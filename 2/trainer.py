import torch
from torch.nn import Module
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F

from argparse import Namespace
from tqdm import tqdm
# from datasets import BowDataset, EmbeddingsDataset, Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import logging
from evaluation import evaluate_batch
from pathlib import Path

from typing import Callable, Dict

import os

from datetime import datetime as dt

class Trainer:
    """Generic trainer class for pytorch models."""
    def __init__(self, model: Module, train_iter: DataLoader, val_iter: DataLoader, 
                epochs: int, lr: float, loss_fn: Callable, model_args:Dict):
        """
        
        """
        self.logger = logging.getLogger(__name__)

        # setting up checkpoint storage
        time = dt.now()
        # timestamp string with date followed by HHmm wallclock time (used to name checkpoint dirs)
        timestamp = f"{time.date()}_{time.hour:0>2}{time.minute:0>2}"
        self.checkpoint_dir = Path("models")/timestamp
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.model_args = model_args
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.epochs = epochs
        self.loss_f = loss_fn
        self.loss_label = self.loss_f.__name__

        self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=0.0
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                self.epochs * len(train_iter)
            )
        
        self.trained = False # set to true when training is complete

    def train(self, patience:int=4, tol:float=5e-3):
        """
        Trains the model and returns the final loss.

        patience and tol determine early stopping:
            - patience: number of epochs allowed with no (val) loss improvement
            - tol: the tolerated amount of (val) loss degradation between epochs (i.e. without affecting the patience/run counters)
        """
        # variables for determining early stopping
        min_loss = None
        run = patience

        lock = True
        # janky fix to avoid crashing when two training scripts are started at the same time
        while lock:
            try:
                os.mkdir(self.checkpoint_dir)
                lock = False

            except:
                self.checkpoint_dir = self.checkpoint_dir.parent/f"{self.checkpoint_dir.name}x"

        self.logger.info(f"Created checkpoint dir at: '{self.checkpoint_dir}'")

        # for reporting/logging purposes we print details about the setup to a txt file
        model_name = self.model._get_name()
        details_file = self.checkpoint_dir/f"{model_name}.txt"

        with details_file.open("w", encoding="utf-8") as f:
            f.write(
                "\n\n".join([
                    f"Model: {self.model}",
                    f"Loss fn: {self.loss_f.__name__}", # loss fn name,
                    f"Pooling: {"N/A" if model_name == "BaselineModel" else self.model.pooling}", # pooling fn (not sure if explicitly needed)
                    f"Optimizer: {self.optimizer}",
                    f"Scheduler: {type(self.scheduler).__name__}",
                ])
            )

        self.logger.info(f"Starting training.")

        self.history = {
            "train_loss":[],
            "val_loss":[],
            "val_acc":[],
            "val_alpha":[],
            "train_acc":[],
            "train_alpha":[]
        }

        for epoch in range(1,self.epochs+1):
            if not run:
                self.logger.info("Early stopping triggered, hav nice day :)")
                break

            self.logger.info(f"Epoch {epoch}:")

            # for debugging:
            # param norm should be different betweeen/after epochs
            param_norm = torch.sqrt(sum(
                p.data.norm()**2 for p in self.model.parameters() if p.requires_grad
                ))
            self.logger.info(f"Param norm before epoch {epoch}: {param_norm}")

            epoch_preds, epoch_trues, epoch_loss = [], [], []

            self.model.train()
            for batch in tqdm(self.train_iter):

                self.optimizer.zero_grad()

                # inputs, true_labels = (t.to(self.device) for t in batch) # Fetch inputs and gold labels
                true_labels = batch["labels"].to(self.device)

                # if funtuning we need to send all encoded_batch tensors to device
                if "encoded_batch" in batch:
                    batch["encoded_batch"] = {
                        k: (v.to(self.device) if torch.is_tensor(v) else v)
                            for k, v in batch["encoded_batch"].items()
                    }
                    inputs = {k:batch[k] for k in batch if k != "labels"}
                else:
                    inputs = {k:batch[k].to(self.device) for k in batch if k != "labels"}

                pred_labels = self.model(**inputs) # Predict values
                # self.logger.info(f"Pred labels dtype: {pred_labels.dtype}")
                # self.logger.info(f"Pred labels size: {pred_labels.shape}")
                # self.logger.info(f"tRUE labels dtype: {true_labels.dtype}")
                # self.logger.info(f"tRUE labels size: {true_labels.shape}")

                # Ensure shapes are compatible, pred and target shape must be both [batch_size]
                if self.loss_label == "mse_loss":
                    loss = self.loss_f(
                            pred_labels.view(-1),
                            true_labels.view(-1)
                    )
                else:
                    loss = self.loss_f(pred_labels, true_labels)

                loss.backward() # Backward pass

                # for debugging (confirmed that gradient mean is 0.0 for word embedding weights
                # throughout epochs)
                #with torch.no_grad():
                #    for name,p in self.model.named_parameters():
                #        if p.requires_grad and p.grad is not None:
                #            self.logger.info(f"Grad mean {name}: {p.grad.abs().mean().item()}")
                #            break

                self.optimizer.step()
                self.scheduler.step()

                # self.logger.info(f"self.loss_label: {self.loss_label}")
                if self.loss_label != "mse_loss": # classification
                    # convert from 1hot to numeric class labels
                    true = (true_labels.argmax(dim=1)+1).detach().cpu() # +1 accounts for 0 indexing in 1hot labels/preds
                    pred = (pred_labels.argmax(dim=1)+1).detach().cpu()
                else: # regression with mse
                    pred_cont = pred_labels.detach().cpu().view(-1) # continous predictions

                    true = true_labels.detach().cpu().view(-1) # Ensures 1D for tolist() to work
                    pred = torch.clamp(torch.round(pred_cont), 1, 4).cpu() # rounds continous predictions to nearest int in [1,4]

                # add epoch predictions and true labels to lists
                epoch_preds.extend(pred.tolist())
                epoch_trues.extend(true.tolist())

                # store loss for averaging / calculating epoch loss
                epoch_loss.append(loss.detach().cpu())
            # end of batch loop

            param_norm = torch.sqrt(sum(
                p.data.norm()**2 for p in self.model.parameters() if p.requires_grad
                ))
            self.logger.info(f"Param norm after epoch {epoch}: {param_norm}")

            # evaluating on both train- and validation sets
            self.model.eval()
            train_metrics = evaluate_batch(epoch_preds, epoch_trues)
            val_accu, val_alph, val_loss = self.evaluate(use_loss=True)

            # log all metrics (for plotting)
            self.history["train_loss"].append(sum(epoch_loss)/len(self.train_iter)) # average / epoch loss
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_accu) # validation accuracy
            self.history["val_alpha"].append(val_alph) # validation alpha
            self.history["train_acc"].append(train_metrics["accuracy"]) # train accuracy
            self.history["train_alpha"].append(train_metrics["krippendorff_alpha"]) # train accuracy
            self.logger.info(f"train loss : {self.history['train_loss'][-1]:.5f} ; val acc : {self.history['val_acc'][-1]:.5f} ; val alpha : {self.history['val_alpha'][-1]:.5f}")

            # saving and stopping mechanisms go here
            if min_loss is None or min_loss > val_loss:
                # first epoch or current macro avg acc higher than max -> save checkpoint
                self.save_checkpoint()
                self.checkpoint_epoch = epoch
                min_loss = val_loss

                # if we update the checkpoint we also reset the run counter
                run = patience

            elif min_loss+tol < val_loss:
                # if max-tolerance is larger we decrement the run counter
                run -= 1

        # reload the state dict of the best/stopping epoch (in case we want to use the model stored in the Trainer-object for evaluation or something)
        self.model.load_state_dict(self.load_checkpoint(self.checkpoint_dir.relative_to("models"), only_state_dict=True))
        
        self.trained = True
        # NOTE: final metrics for saved model is at ix checkpoint epoch -1 (-1 since epoch numbering starts at 1)
        return self.history['train_loss'][self.checkpoint_epoch-1] # return loss from final checkpoint
    
    def save_checkpoint(self):
        assert self.checkpoint_dir.exists(), "No chekpoint has been created yet;("

        # checkpoint is constructed so that the model object can be re-initialized in the same manner it was first created
        chekpoint = {
            "model":self.model.state_dict(),
            "model_args":self.model_args,
            "model_type":type(self.model)
        }

        torch.save(chekpoint, self.checkpoint_dir/"model.pth")

    @staticmethod
    def load_checkpoint(path:str, only_state_dict:bool=False):
        checkpoint_path = Path("models")/path/"model.pth"

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        if only_state_dict:
            # returns only state dict for loading checkpoint after training
            return checkpoint["model"]

        else:
            model = checkpoint["model_type"](checkpoint["model_args"])

            model.load_state_dict(checkpoint["model"])

            return model

    def plot_histroy(self, out_name:str, history:dict=None):
        assert self.trained, "Model must be trained first"

        if history is None:
            history = self.history

        _, axes = plt.subplots(1,3,figsize=(18,6))

        epochs = len(self.history["train_loss"]) # choice of key is arbitrary

        x = np.arange(epochs)+1 # +1 accounts for our epoch loop counting from 1
        tloss_y = history["train_loss"]
        vloss_y = history["val_loss"]

        axes[0].plot(x, tloss_y, color="tab:cyan", label="train")
        axes[0].plot(x, vloss_y, color="tab:pink", label="val", linestyle="-.")
        axes[0].axvline(x=self.checkpoint_epoch, color="red", linestyle="--", alpha=0.7)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch #")
        axes[0].set_ylabel(self.loss_label)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()


        valpha_y = history["val_alpha"]
        talpha_y = history["train_alpha"]

        axes[1].plot(x, valpha_y, color="tab:blue", label="val")
        axes[1].plot(x, talpha_y, color="tab:orange", label="train", linestyle="-.")
        axes[1].axvline(x=self.checkpoint_epoch, color="red", linestyle="--", alpha=0.7)
        axes[1].set_title("Alpha")
        axes[1].set_xlabel("Epoch #")
        axes[1].set_ylabel("Krippendorff's alpha")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()        
        
        
        vacc_y = history["val_acc"]
        tacc_y = history["train_acc"]

        axes[2].plot(x, vacc_y, color="tab:green", label="val")
        axes[2].plot(x, tacc_y, color="tab:purple", label="train", linestyle="-.")
        axes[2].axvline(x=self.checkpoint_epoch, color="red", linestyle="--", alpha=0.7)
        axes[2].set_title("Accuracy")
        axes[2].set_xlabel("Epoch #")
        axes[2].set_ylabel("Accuracy")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.suptitle(f"Training stats: {out_name}")
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir/f"{out_name}.pdf")

    @torch.no_grad()
    def evaluate(self, use_loss: bool=False):
        """
        Returns average validation accuracy and krip alpha over b batches.
        -> val_acc, val_alpha
        -> val_acc, val_alpha, val_loss (if loss)
        """
        all_preds, all_trues = [], []
        val_loss = []

        self.model.eval()
        for batch in self.val_iter:

            trues = batch["labels"].to(self.device)

            # if funtuning we need to send all encoded_batch tensors to device except sequence IDs
            if "encoded_batch" in batch:
                batch["encoded_batch"] = {
                    k: (v.to(self.device) if torch.is_tensor(v) else v) # Ignores sequence IDs
                    for k,v in batch["encoded_batch"].items()
                }
                inputs = {k:batch[k] for k in batch if k != "labels"}
            else:
                inputs = {k:batch[k].to(self.device) for k in batch if k != "labels"}

            preds = self.model(**inputs)

            if use_loss:
                if self.loss_label == 'mse_loss':
                    loss = self.loss_f(preds.view(-1), trues.view(-1))
                else:
                    loss = self.loss_f(preds, trues)
                val_loss.append(loss.detach().cpu())
            
            if self.loss_label != 'mse_loss': # classificaiton
                true_labels = (trues.argmax(dim=1)+1).detach().cpu() # +1 accounts for 0 indexing in 1hot labels/preds
                pred_labels = (preds.argmax(dim=1)+1).detach().cpu()
            else: # regression
                true_labels = trues.detach().cpu().view(-1) # Reshape to 1D for regression
                pred_labels = torch.clamp(
                    torch.round(preds), # round to nearest label for mse
                    1,
                    4 # Clamping to always be between 1 and 4 for correct labels
                )
            

            all_preds.extend(pred_labels.tolist())
            all_trues.extend(true_labels.tolist())

        metrics = evaluate_batch(all_preds, all_trues)

        if use_loss:
            final_loss = sum(val_loss)/len(self.val_iter)

            return metrics["accuracy"], metrics["krippendorff_alpha"], final_loss
        
        else:
            return metrics["accuracy"], metrics["krippendorff_alpha"]

if __name__ == "__main__":
    pass
