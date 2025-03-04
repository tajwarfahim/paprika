from typing import (
    Dict,
    Union,
    Tuple,
    Tuple,
    Any,
    Optional,
)
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class OfflineRWRTrainer(Trainer):
    """
    Trainer for Offline RWR.

    Loss function = (log pi(a|s)) * exp(r(s, a)/beta)
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of this class.
        Special inputs besides the regular one that the parent class takes:
            rwr_temperature (float):
                the beta parameter in the loss function above.
                Default: 1.0

            rwr_type (str):
                The type of the reward weighting done.
                Default: "exp"

            rwr_term_max (float):
                The max value that we clamp the reward weight to.
                Default: 3.0

            rwr_term_min (float):
                The min value that we clamp the reward weight to.
                Default: 0.0
        """
        self.rwr_temperature = kwargs.pop("rwr_temperature", 1.0)
        self.rwr_type = kwargs.pop("rwr_type", "exp")
        self.rwr_term_max = kwargs.pop("rwr_term_max", 3.0)
        self.rwr_term_min = kwargs.pop("rwr_term_min", 0.0)

        super().__init__(*args, **kwargs)

    def get_rwr_term(
        self,
        rewards: torch.Tensor,
        rwr_type: str,
    ) -> torch.Tensor:
        """
        Calculates the the reward weight for RWR.

        Input:
            rewards (torch.Tensor):
                The rewards per token, coming from the dataset.

            rwr_type (str):
                The type of the reward weight term.
                Default: "exp"

        Output:
            rwr_term (torch.Tensor):
                Per token RWR weight term
        """
        if rwr_type == "exp":
            return torch.exp(rewards / self.rwr_temperature)
        elif rwr_type == "baseline_batch_mean":
            return rewards - rewards.mean()
        elif rwr_type == "baseline_ema_batch_mean":
            raise NotImplementedError
        elif rwr_type == "baseline_running_mean":
            raise NotImplementedError
        elif rwr_type == "rloo":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"RWR type {rwr_type} not implemented")

    def compute_loss(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tuple[Any, torch.Tensor], torch.Tensor]:
        """
        Computes and returns offline reward weighted regression (RWR) loss.
        Offline RWR loss function:
            L_RWR = - log(π(a|s)) * exp(r(s,a)/β)

        Input:
            model (PreTrainedModel or torch.nn.Module):
                A language model that we are trying to finetune.

            inputs (Dict[str, torch.Tensor]):
                The inputs to the language model. This needs to contain
                "labels" and "input_ids" as keys, and corresponding tensors.

            return_outputs (bool):
                whether to return the loss only, or to return the model outputs
                as well.
                Default: False

            num_items_in_batch (int):
                Number of items in the batch
                Default to None

                NOTE: this has been included so that the code can be used with the
                most recent version of transformers

        Output:
            If return_outputs is False, then loss (torch.Tensor)
            If return_outputs is True, then
                (model_outputs (Any), loss (torch.Tensor)) Tuple
        """

        labels = inputs["labels"]
        rewards = inputs.pop("rewards", torch.zeros_like(labels))  # shape should be (B, T)
        model_output = model(
            **inputs
        )  # Standard forward pass, for testing purposes we can use the loss term from here.

        if self.args.past_index >= 0:
            self._past = model_output[self.args.past_index]

        model_name = self.accelerator.unwrap_model(model)._get_name()

        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            shift_labels = True

        # Logits shape: (B, T, V); B - Batch size, T - Sequence length, V - Vocabulary size
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        # This is for loss computation with the
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            rewards = rewards[
                ..., 1:
            ].contiguous()  # Shift rewards to the same extent as the labels

        # This is just cross-entropy; but weighted by the RWR term
        log_probs = -F.log_softmax(logits, dim=-1)  # Log softmax over the vocabulary dimension
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(
                -1
            )  # Adds a dimension to make it broadcastable for the vocabulary dimension?
        if rewards.dim() == log_probs.dim() - 1:
            rewards = rewards.unsqueeze(
                -1
            )  # Adds a dimension to make it broadcastable for the vocabulary dimension?

        padding_mask = labels.eq(IGNORE_TOKEN_ID)  # B x T
        # Makes the -100 to 0;
        labels = torch.clamp(labels, min=0)
        # Gather the log probs at the label indices over the vocabulary dimension
        nll_loss = log_probs.gather(dim=-1, index=labels)
        nll_loss.masked_fill_(padding_mask, 0.0)
        # Multiply by the RWR term; You dont calculate the loss on any of the masked terms.
        nll_loss = nll_loss * torch.clamp(
            self.get_rwr_term(rewards, self.rwr_type),
            min=self.rwr_term_min,
            max=self.rwr_term_max,
        )
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        # Average over only the non-masked elements
        nll_loss = nll_loss.sum() / num_active_elements
        return (nll_loss, model_output) if return_outputs else nll_loss
