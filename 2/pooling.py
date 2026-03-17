import torch
from transformers import AutoTokenizer
from typing import Any, List, Dict
import logging    

def bos_pooling(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Takes in 2D tensor of sentences and returns the CLS/BOS-token of each sentence as a 1D tensor
    """
    # Embeddings: [batch_size, seq_len, hidden_size]
    return embeddings[:, 0, :]

def target_word_pooling(embeddings: torch.Tensor, instances: List[Dict], encoded_batch: Dict[str, Any]) -> torch.Tensor:
    """
    Takes in a batch of embeddings and sentence pair instances
    Returns a list of the representations of their target words concatenated
    """
    
    logger = logging.getLogger(__name__)
    all_pooled = []

    for i, inst in enumerate(instances):
        # logger.info(f"instance: {inst}")
        spans = [inst["word_indices_a"], inst["word_indices_b"]]
        sents = [inst["sentence_a"], inst["sentence_b"]]

        # List of offsets
        offset_mapping = encoded_batch["offset_mapping"][i].tolist()

        # List of 0, 1 or None telling which of the two sentences each token belongs to
        # Sentence A -> sequence_id == 0
        # Sentence B -> sequence_id == 1
        # Special tokens [CLS], [SEP] etc. -> sequence_id == None
        sequence_ids = encoded_batch["sequence_ids"][i]

        # For storing the final representation of the target word
        sent_vecs = []

        # Find all token indices in this instance that whose char offsets fall within this given span
        for sent_id, span in enumerate(spans): # sent_id = 0 for sent A, 1 for sent B
            span_start, span_end = span
            token_indices = []

            for j, (start, end) in enumerate(offset_mapping):
                # Skip special tokens, which should be (0,0)
                if start == 0 and end == 0:
                    continue

                # Skip tokens from the other sentence
                if sequence_ids[j] != sent_id:
                    continue
                
                sent = sents[sent_id]
                    
                # Pick out relevant subword token indices that land within the span
                # The tokenizer spans include leading white spaces while instance spans do not.
                # Therefore, first check if first character is space and append if relevant
                if sent[start] == " " and (start >= span_start - 1) and (end <= span_end):
                    token_indices.append(j)
                    continue

                # Span does not start with leading space, append if relevant
                if start >= span_start and end <= span_end:
                    token_indices.append(j)

            if not token_indices:
                # Did not find any token indices for some reason 
                # if sent_id == 0: 
                #     # print(f"Did not find any token indices of {inst['word']} for {inst["sentence_a"]}")  
                #     pass
                # else:
                #     # print(f"Did not find any token indices of {inst['word']} for {inst["sentence_b"]}")  
                #     pass
                avg_emb = torch.zeros(embeddings.shape[2], device=embeddings.device) # shape[2] = hidden_size?
            else:
                # Gather embeddings for the relevant indices
                # Picks i-th example in the batch, the rows in token_indices and all : hidden dimensions
                token_embs = embeddings[i, token_indices, :] # [n_subwords, hidden_size]
                # Find the average
                avg_emb = token_embs.mean(dim=0)
            
            sent_vecs.append(avg_emb)

        # Concatenate the two sentences' representations
        pooled = torch.cat(sent_vecs, dim=0) # [hidden_size + hidden_size]
        all_pooled.append(pooled)

    # Make a stack over the whole batch
    pooled_targets = torch.stack(all_pooled, dim=0)
    return pooled_targets 

# For debugging purposes
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    sent_pairs = [("I love ludvig", "Ludvig loves cucumber")]
    instances = [
        {
            "sentence_a" : "Hey Ludvig",
            "sentence_b" : "Ludvig loves cucumber",
            "word": "Ludvig",
            "word_indices_a" : [4, 10],
            "word_indices_b" : [0, 6]
        },
        {
            "sentence_a" : "I love fish",
            "sentence_b" : "fish loves me",
            "word": "fish",
            "word_indices_a" : [7, 11],
            "word_indices_b" : [0, 4]
        }
    ]

    sentences_a = [inst["sentence_a"] for inst in instances]
    sentences_b = [inst["sentence_b"] for inst in instances]

    encoded_batch = tokenizer(
        sentences_a,
        sentences_b,
        return_offsets_mapping=True,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    print(encoded_batch)

    batch_size = encoded_batch["input_ids"].shape[0]
    seq_len = encoded_batch["input_ids"].shape[1]
    hidden_size = 2
    # Fake embeddings
    embeddings = torch.randn(batch_size, seq_len, hidden_size)
    # print(f"Embeddings: {embeddings}")


    pooled = target_word_pooling(embeddings=embeddings, encoded_batch=encoded_batch, instances=instances)

    # print("Embeddings shape:", embeddings.shape)   # [batch_size, seq_len, hidden_size]
    # print("Pooled shape:", pooled.shape)           # [batch_size, 2 * hidden_size]
    # print("All pooled:")
    # print(pooled)
