from typing import (
    Dict,
    Any,
    Tuple,
    Union,
    Optional,
)
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class GEMTrainer(Trainer):
    """
    Uses GEM loss instead of CE to run SFT on a language model.
    GEM loss in introduced in this paper:
    https://arxiv.org/abs/2408.16673v1
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of this trainer class.

        Special inputs besides the regular one that the parent class takes:
            beta (float):
                The beta parameter in the GEM loss
                Default: 0.7

            h_function (str):
                Which h function to use in the loss calculation.
                Valid choices: "linear", "log_sigmoid"
                Default: "linear"
        """
        self.beta = kwargs.pop("beta", 0.7)
        self.h_function = kwargs.pop("h_function", "linear")
        assert self.h_function in ["linear", "log_sigmoid"]

        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tuple[Any, torch.Tensor], torch.Tensor]:
        """
        Computes and returns loss, defined in the GEM paper.
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
        model_output = model(**inputs)

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

        loss = self.calculate_GEM_loss(logits=logits, labels=labels)
        if return_outputs:
            return (loss, model_output)
        else:
            return loss

    def calculate_GEM_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the GEM loss based on this paper:
        https://arxiv.org/abs/2408.16673v1

        Input:
            logits (torch.Tensor):
                logits outputted from the model.

                Shape: (B, T, V), where:
                    B = batch size,
                    T = sequence length
                    V = vocabulary size

            labels (torch.Tensor):
                The actual tokens that we want to regress to.
                Some of these tokens have label -100 (or more generall IGNORE_TOKEN_ID)
                we ignore these tokens from the loss calculation.

                Shape: (B, T), where:
                    B = batch size
                    T = sequence length

        Output:
            loss (torch.Tensor):
                The calculated loss.
        """
        mask = labels != IGNORE_TOKEN_ID
        logits = logits[mask]
        labels = labels[mask]

        with torch.no_grad():
            logits_on_labels = torch.gather(
                logits, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)

            logits_diff = logits - logits_on_labels.unsqueeze(-1)

            if self.h_function == "linear":
                weights = torch.ones_like(logits_diff)
            elif self.h_function == "log_sigmoid":
                weights = F.sigmoid(0.01 * logits_diff)
            else:
                raise ValueError(f"Given h function {self.h_function} is not supported.")

        gene_log_probs = F.log_softmax(logits, dim=-1)
        q_probs = torch.exp(F.log_softmax(logits / self.beta, dim=-1)).detach()

        real_log_probs = torch.gather(
            gene_log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        loss = -torch.sum(
            q_probs * weights * (real_log_probs.unsqueeze(-1) - gene_log_probs), dim=-1
        ).mean()

        return loss
