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
from torch.utils.data import DataLoader
from data.data_loading import Utils
from data.datasets import BaselineDataset
import argparse
import torch.nn.functional as F
import logging

from model import BaselineModel as Model

from trainer import Trainer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Baseline model Oblig 2 IN5550.')

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
        default='/fp/projects01/ec403/IN5550/obligatories/2/en_train.jsonl.gz', 
        help='Path to training data.')
    parser.add_argument(
        '--val_path', 
        type=str, 
        default='/fp/projects01/ec403/IN5550/obligatories/2/en_dev.jsonl.gz', 
        help='Path to validation data.')
    parser.add_argument(
        '--emb_path',
        type=str,
        default='/fp/projects01/ec403/models/static/82/model.bin',
        help='Path to static embeddings model.')
    parser.add_argument(
        '-l', '--local',
        action="store_true",
        help='For running the script locally.'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    args = parse_arguments()
    
    torch.manual_seed(args.seed)

    loss = F.cross_entropy
    
    if args.local:
        embs = load_embedding("../localdata/embeddings/82/model.bin")

        train = Utils.load_json("../localdata/en_train.jsonl.gz")
        val = Utils.load_json("../localdata/en_dev.jsonl.gz")

    else:
        embs = load_embedding(args.emb_path)
        # embs = Utils.add_special_tokes(embs) # maybe not needed ?

        train = Utils.load_json(args.train_path)
        val = Utils.load_json(args.val_path)

    train_data = BaselineDataset(train,embs)
    val_data = BaselineDataset(val,embs)

    model_args = {
        "args":args,
        "input_dim":embs.vector_size,
        "n_classes":4
    }

    model = Model(**model_args)

    train_iter = DataLoader(train_data, args.batch_size)
    val_iter = DataLoader(val_data, args.batch_size)

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

    trainer.plot_histroy("baseline")

    val_acc, val_alpha = trainer.evaluate()
    logger.info("Evaluation metrics on validation set:")
    logger.info(f"Accuracy: {val_acc:.5f}")
    logger.info(f"Krippendorff's alpha: {val_alpha:.5f}")
