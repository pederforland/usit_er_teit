import torch
import torch.nn as nn
import argparse
import logging
import torch.nn.functional as F
from transformers import AutoModel
from typing import List, Dict
from argparse import Namespace
from pooling import bos_pooling, target_word_pooling

class BaselineModel(nn.Module):
    """
    Based on MLP/FFNN regressor developed for Oblig1.
    """
    def __init__(self, args: argparse.Namespace, input_dim: int, n_classes: int):
        super().__init__()
        self.args = args

        # Initialize layer list with first layer
        self.layers = nn.ModuleList(
            # Input layer
            [nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(input_dim, args.hidden_size),
                nn.ReLU()
            )] 
            +
            # Hidden layers
            [nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU()
            ) 
            for i in range(args.num_hidden)
            ]
            +
            # Output layer
            [nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, n_classes),
                # nn.Softmax(dim=1) # cannot be softmax if MSE
            )]
        )

    def forward(self, x):
        output = x
        for layer in self.layers: # Passes x through all layers of the model
            output = layer(output)
        return output.squeeze(-1)

class WiCModel(nn.Module):
    def __init__(self, args: Namespace, n_classes: int, input_dim: int = None):
        super().__init__()
        if args.local == 1:
            # Get model locally
            self.encoder = AutoModel.from_pretrained(args.encoder)
        else:
            # Running on fox, get model from fox
            self.encoder = AutoModel.from_pretrained(f"/fp/projects01/ec403/hf_models/{args.encoder}")

        hidden_size = self.encoder.config.hidden_size
        self.pooling = args.pooling # Str
        self.logger = logging.getLogger(__name__)
        # MLP over the concatenated target-word representation
        if args.pooling == "BOS":
            input_dim = hidden_size
        else:
            input_dim = 2 * hidden_size 

        self.classifier = BaselineModel(args=args, input_dim=input_dim, n_classes=n_classes)

    def forward(self, encoded_batch: Dict[str, torch.Tensor], instances: List[Dict]):
        """
        encoded_batch: output of tokenizer(sentences_a, sentences_b, ...),
                       must include "input_ids", "attention_mask",
                       and "offset_mapping"
        instances: list of metadata dicts aligned with batch

        returns: logits [batch_size, num_labels]
        """
        # Making encoder kwargs wauw
        encoder_kwargs = {
            "input_ids": encoded_batch["input_ids"],
            "attention_mask": encoded_batch["attention_mask"],
        }

        # token_type_ids is not used by mmBERT-small so do a check if we should pass it or not
        # (this is why we have to do the kwargs route)
        # type_vocab_size > 1 is apparently the common check for this
        if "token_type_ids" in encoded_batch and getattr(self.encoder.config, "type_vocab_size", 0) > 1:
            encoder_kwargs["token_type_ids"] = encoded_batch["token_type_ids"]

        # Run encoder
        outputs = self.encoder(**encoder_kwargs)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Pooling now yees
        if self.pooling == "BOS":
            pooled = bos_pooling(embeddings)
        else:
            pooled = target_word_pooling(embeddings=embeddings, encoded_batch=encoded_batch, instances=instances)
        # pooled: [batch_size, 2 * hidden_size]

        logits = self.classifier(pooled)       # [batch_size, num_labels]
        return logits

"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")

def collate_fn(batch):
    # batch is a list of instances (dicts with sentence_a, sentence_b, etc.)
    sentences_a = [inst["sentence_a"] for inst in batch]
    sentences_b = [inst["sentence_b"] for inst in batch]

    encoded_batch = tokenizer(
        sentences_a,
        sentences_b,
        return_offsets_mapping=True,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Extract labels from instances
    labels = torch.tensor([inst["median_label"] for inst in batch], dtype=torch.long)

    return encoded_batch, batch, labels
    # encoded_batch: dict of tensors (input_ids, attention_mask, token_type_ids, offset_mapping)
    # batch: the original list of instances (for spans)
    # labels: tensor [batch_size]

***Training loop sketch:***

from torch.utils.data import DataLoader

dataset = ...  # your WiC dataset as a list or custom Dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

model = WiCTargetPoolingModel(model_name="jhu-clsp/mmBERT-base", num_labels=5)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(num_epochs):
    for encoded_batch, instances, labels in dataloader:
        # Move tensors to device
        encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(encoded_batch, instances)       # [batch_size, num_labels]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

"""
