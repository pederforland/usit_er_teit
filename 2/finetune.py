"""
Thoughts:
- static embeddings should struggle since the task relies on *meaning in context*
    - static word embeddings only model one "meanin" per word, i.e., one static embedding for each word $w\\in{}V$
    - contextualized models address this by also modeling the context of the word in the embedding, resulting in more nuanced embeddings

- static embeddings directory on fox:
    - /fp/projects01/ec403/models/static

- dataset directory on fox:
    - /fp/projects01/ec403/IN5550/obligatories/2/en_train.jsonl.gz
"""

from load_embedding import load_embedding
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from data.data_loading import Utils, CollateFunctor
from data.datasets import FinetuneDataset
from transformers import AutoTokenizer
from loss import emd_square

# typing
from typing import List, Dict, Literal
import argparse
import torch.nn.functional as F

import logging

from model import WiCModel as Model

from trainer import Trainer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Finetune model Oblig 2 IN5550.')

    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=128, 
        help='The batch size.')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate.')
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10, 
        help='The number of epochs to train.')
    parser.add_argument(
        '--hidden_size', 
        type=int, 
        default=128, 
        help='Dimension of the hidden layers')
    parser.add_argument(
        '--num_hidden', 
        type=int, 
        default=2, 
        help='Number of hidden layers.')
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-4, 
        help='The learning rate.')
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='The random seed.')
    parser.add_argument(
        '--train_path', 
        type=str, 
        default='/fp/projects01/ec403/IN5550/obligatories/2/en_train.jsonl.gz', # make this smarter by assuming we already are in the correct directory (what if I don't want to though)
        help='Path to training data.')
    parser.add_argument(
        '--val_path', 
        type=str, 
        default='/fp/projects01/ec403/IN5550/obligatories/2/en_dev.jsonl.gz', 
        help='Path to validation data.')
    parser.add_argument(
        '--loss',
        type=str,
        default='MSE',
        help='Either MSE, CE, EMD_median or EMD_full'
    )
    parser.add_argument(
        '--pooling',
        type=str,
        default='BOS',
        help='Either BOS or target_word'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='bert-base-cased',
        help='Path to encoder model'
    )
    parser.add_argument(
        '--local',
        type=int,
        default=0,
        help='Option to run locally. Has to be used with your own encoder argument stored locally'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    args = parse_arguments()
    
    torch.manual_seed(args.seed)

    loss_fns = {
        'MSE' : F.mse_loss,
        'CE' : F.cross_entropy,
        'EMD_median' : emd_square,
        'EMD_full' : emd_square
    }
    loss = loss_fns[args.loss]

    train = Utils.load_json(args.train_path)
    val = Utils.load_json(args.val_path)

    train_data = FinetuneDataset(train)
    val_data = FinetuneDataset(val)
    if args.local == 1:
        # Get tokenizer locally
        tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    else:
        # get tokenizer from fox
        tokenizer = AutoTokenizer.from_pretrained(f"/fp/projects01/ec403/hf_models/{args.encoder}")
    collate_fun = CollateFunctor(tokenizer, loss=args.loss)
    n_classes = 1 if args.loss == 'MSE' else 4

    model_args = {
        'args' : args,
        'n_classes' : n_classes
    } # Makes saving checkpoitns easiers
    model = Model(**model_args)

    # verify that we actually have any trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameter count: {sum(p.numel() for p in params)}")

    train_iter = DataLoader(train_data, args.batch_size, collate_fn=collate_fun)
    val_iter = DataLoader(val_data, args.batch_size, collate_fn=collate_fun)


    trainer = Trainer(
        model=model,
        model_args=model_args,
        train_iter=train_iter,
        val_iter=val_iter,
        epochs=args.epochs,
        lr=args.lr,
        loss_fn=loss
    )

    trainer.train()

    trainer.plot_histroy("finetune") # plot MY troy?!

    val_acc, val_alpha = trainer.evaluate()
    logger.info("Evaluation metrics on validation set:")
    logger.info(f"Accuracy: {val_acc:.5f}")
    logger.info(f"Krippendorff's alpha: {val_alpha:.5f}")
