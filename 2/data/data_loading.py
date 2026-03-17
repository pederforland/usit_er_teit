import gzip
import json
from pathlib import Path
import torch
from gensim.models import KeyedVectors
from typing import List, Dict
from transformers import AutoTokenizer
from torch.nn.functional import one_hot
import logging

class Utils:
    @staticmethod
    def add_special_tokes(word2vec:KeyedVectors) -> KeyedVectors:
        """Use to add special tokens [UNK] and [PAD] to pretrained w2v models"""
        word2vec["[UNK]"] = torch.tensor(word2vec.vectors).mean(dim=0).numpy()
        word2vec["[PAD]"] = torch.zeros(word2vec.vector_size).numpy()
        
        return word2vec

    @staticmethod
    def load_json(path:str) -> List[Dict]:
        """
        Loads a given json file:
        determines if decompression is neccessary depending on extension.
        """
        path = Path(path)

        if path.suffix == ".gz":
            f = gzip.open(path, "rt", encoding="utf-8")
        else:
            f = open(path, "r", encoding="utf-8")

        data = [json.loads(l) for l in f]

        f.close()

        return data

    @staticmethod
    def to_dist(labels: List[List], dtype=torch.int64):
        """Converts a list of label judgments (variable length) to distributions of length 4"""
        t = torch.tensor(labels, dtype=dtype) - 1  # shift labels 1-4 to 0-3
        counts = torch.bincount(t, minlength=4) # count each index
        return counts.float() / counts.sum() # normalise
    
class CollateFunctor:
    def __init__(self, tokenizer, loss:str, max_length:int=512):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer
        self.loss = loss

    def __call__(self, batch):
        # batch is a list of instances (dicts with sentence_a, sentence_b, etc.)
        sentences_a = [inst["sentence_a"] for inst in batch]
        sentences_b = [inst["sentence_b"] for inst in batch]
        labels = [inst["labels"] if self.loss == 'EMD_full' else inst["median_label"] for inst in batch]

        encoded_batch = self.tokenizer(
            sentences_a,
            sentences_b,
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Store sequence IDs to make target word pooling easier later down the line
        sequence_ids = [encoded_batch.sequence_ids(i) for i in range(len(sentences_a))]
        encoded_batch = dict(encoded_batch) # convert to plain dict for DataLoader
        encoded_batch["sequence_ids"] = sequence_ids

        # Extract labels from instances
        if self.loss == 'MSE':
            print(f"CollateFunctor: self.loss={self.loss}")
            labels = torch.tensor(labels, dtype=torch.float32).squeeze(-1)
        elif self.loss == 'EMD_full':
            labels = torch.stack([Utils.to_dist(l) for l in labels])
        else:
            labels = one_hot(torch.LongTensor(labels))
        # labels = torch.tensor([inst["target"] for inst in batch], dtype=torch.long)

        # self.logger.info(f"Labels: {labels}")
        # self.logger.info(f"Labels dtype: {labels.dtype}")

        # batch: the original list of instances (for spans)
        # encoded_batch: dict of tensors (input_ids, attention_mask, token_type_ids, offset_mapping)
        # labels: tensor [batch_size]

        # self.logger.info(f"Labels: {labels}")

        return {"instances": batch, "encoded_batch": encoded_batch, "labels": labels}
