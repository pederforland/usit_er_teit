import torch

def emd_square(pred_dist: torch.Tensor, gold_labels: torch.Tensor) -> torch.Tensor:
    """gold_labels must be of shape [batch_size, 4]. Either distribution or one-hot. pred_dist is a
    predicted distribution from the model, where Softmax has been applied."""

    assert pred_dist.size() == gold_labels.size(), f"pred and gold labels must be of same shape. gold: {gold_labels.size()}, pred: {pred_dist.size()}"
    assert gold_labels.size(dim=1) == 4, f"Dimension of labels must be 4 (dist or one-hot). Got shape {gold_labels.size()}"

    cdf_pred = torch.cumsum(pred_dist, dim=1)
    cdf_gold = torch.cumsum(gold_labels, dim=1)

    return torch.mean(torch.sum((cdf_pred - cdf_gold) ** 2, dim=1) / gold_labels.shape[1])
