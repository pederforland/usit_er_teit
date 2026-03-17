from typing import List, Dict
from torch.utils.data import Dataset
import torch
import re
from gensim.models import KeyedVectors
from string import punctuation
from .data_loading import Utils
from transformers import AutoTokenizer

class BaselineDataset(Dataset):
    PUNCTS = f"[{re.escape(punctuation)}]" # punctuation pattern used by self.__getitem__

    def __init__(self, data:List[Dict], embeddings:KeyedVectors, n:int=4, lower:bool=True):
        self.data = data
        self.embeddings = embeddings
        self.n = n # context window
        self.lower = lower # should ideally be based on the embeddings model used

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        inst = self.data[ix]

        target = inst["median_label"]
        word = inst["word"].lower() if self.lower else inst["word"]
        keys = [word]

        # extracts all surrounding words from sents a and b
        for sent,span in [(inst["sentence_a"],inst["word_indices_a"]), (inst["sentence_b"],inst["word_indices_b"])]:
            if self.lower:
                sent = sent.lower()

            # simple tokenization -> list of separated words and punctuation
            # words = re.findall(r"\w+|"+self.PUNCTS, sent)
            before = re.findall(r"\w+|"+self.PUNCTS, sent[:span[0]])
            after = re.findall(r"\w+|"+self.PUNCTS, sent[span[1]:])

            before.reverse()

            keys += before[:self.n+1]
            keys += after[:self.n+1]

        # use KeyedVectors built in method for getting mean embedding
        x = self.embeddings.get_mean_vector(keys)

        one_hot_target = torch.zeros(4, dtype=torch.double)

        one_hot_target[target-1] = torch.tensor(1, dtype=torch.double)


        return {"x":torch.tensor(x), "labels":one_hot_target}

class FinetuneDataset(Dataset):
    def __init__(self, data:List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        inst = self.data[ix]
        sentence_a = inst["sentence_a"]
        sentence_b = inst["sentence_b"]
        labels = inst["labels"]
        median_label = inst["median_label"]
        word_indices_a = inst["word_indices_a"]
        word_indices_b = inst["word_indices_b"]

        return {"sentence_a": sentence_a, "sentence_b": sentence_b, "labels": labels, "median_label": median_label, "word_indices_a": word_indices_a, "word_indices_b" : word_indices_b}
