import torch
import torch.nn.functional as F

from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def get_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
    token_level: bool = False,
) -> torch.Tensor:
    """
    Given logits and labels, we calculate the log probs (per token, per label)

    Input:
        logits (torch.Tensor):
            logits from the model

            Shape: (batch size, sequence length - 1, vocabulary size)

        labels (torch.Tensor):
            Per token labels

            Shape: (batch size, sequence length - 1)

        average_log_prob (bool):
            Whether we average, per non-masked token, the calculated log probabilties

        token_level (bool):
            Whether to return per token log probs (without
            ignoring/masking the IGNORE_TOKEN_ID tokens)

    Outputs:
        log_probs (torch.Tensor):
            The log probabilities calculated from logits and labels
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels.clone()
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != IGNORE_TOKEN_ID
    labels[labels == IGNORE_TOKEN_ID] = 0

    log_probs = torch.gather(
        log_probs,
        dim=2,
        index=labels.unsqueeze(2),
    ).squeeze(2)

    if token_level:
        return log_probs * mask
    elif average_log_prob:
        return (log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
    else:
        return (log_probs * mask).sum(dim=-1)
