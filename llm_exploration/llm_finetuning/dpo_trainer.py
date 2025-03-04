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
from transformers import (
    Trainer,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_pt_utils import LabelSmoother

from llm_exploration.llm_finetuning.trainer_utils import get_log_probs


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class DPOTrainer(Trainer):
    """
    A minimal implementation for DPO in the multi-turn setting.

    DPO paper: https://arxiv.org/abs/2305.18290
    Q^* paper: https://arxiv.org/abs/2404.12358
    Agent-Q paper: https://arxiv.org/abs/2408.07199
    RPO paper: https://arxiv.org/abs/2404.19733

    Simple case, we have a winning trajectory and a losing one.
    For each trajectory, we just sum up the losses per assistant tokens.

    NOTE: To save memory, we modify the basic DPO implementation to
    precompute ref log probs, so we don't have to store the reference model
    during training.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of this trainer class.

        Special inputs besides the regular one that the parent class takes:
            beta (float):
                The beta parameter in the DPO loss
                Default: 0.1

            rpo_alpha (float):
                The alpha parameter for RPO loss
                If rpo_alpha > 0, then the loss is computed as:
                    Loss = DPO loss + rpo_alpha * NLL loss on chosen sequences
        """
        self.beta = kwargs.pop("beta", 0.1)
        self.rpo_alpha = kwargs.pop("rpo_alpha", 0.0)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tuple[Any, torch.Tensor], torch.Tensor]:
        """
        Computes and returns loss, defined in the DPO paper.
        Input:
            model (PreTrainedModel or torch.nn.Module):
                A language model that we are trying to finetune.

            inputs (Dict[str, torch.Tensor]):
                The inputs to the language model. This needs to have
                the following shape:

                {
                    "input_ids": chosen_input_id (tensor),
                    "labels": chosen_label (tensor),
                    "labels_sft": chosen_label_sft (tensor),
                    "attention_mask": chosen_attention_mask (tensor),
                    "ref_log_probs": chosen_ref_log_probs (tensor),
                    "rejected_input_ids": rejected_input_id (tensor),
                    "rejected_labels": rejected_label (tensor),
                    "rejected_attention_mask": rejected_attention_mask (tensor),
                    "rejected_ref_log_probs": rejected_ref_log_probs (tensor),
                }

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
        chosen_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        chosen_model_outputs = model(**chosen_inputs)
        chosen_logits = (
            chosen_model_outputs["logits"]
            if isinstance(chosen_model_outputs, dict)
            else chosen_model_outputs[0]
        )

        rejected_inputs = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"],
        }
        rejected_model_outputs = model(**rejected_inputs)
        rejected_logits = (
            rejected_model_outputs["logits"]
            if isinstance(rejected_model_outputs, dict)
            else rejected_model_outputs[0]
        )

        model_name = self.accelerator.unwrap_model(model)._get_name()

        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            shift_labels = True
        else:
            raise ValueError(f"The case of shift_labels = False is not supported.")

        chosen_labels = inputs["labels"]
        rejected_labels = inputs["rejected_labels"]
        chosen_labels_sft = inputs["labels_sft"]
        chosen_ref_log_probs = inputs["ref_log_probs"]
        rejected_ref_log_probs = inputs["rejected_ref_log_probs"]

        if shift_labels:
            chosen_labels = chosen_labels[:, 1:]
            rejected_labels = rejected_labels[:, 1:]
            chosen_labels_sft = chosen_labels_sft[:, 1:]

            chosen_logits = chosen_logits[:, :-1, :]
            rejected_logits = rejected_logits[:, :-1, :]

        loss = self.calculate_dpo_loss(
            chosen_logits=chosen_logits,
            chosen_ref_log_probs=chosen_ref_log_probs,
            chosen_labels=chosen_labels,
            chosen_labels_sft=chosen_labels_sft,
            rejected_logits=rejected_logits,
            rejected_ref_log_probs=rejected_ref_log_probs,
            rejected_labels=rejected_labels,
        )

        if return_outputs:
            return (loss, chosen_model_outputs)
        else:
            return loss

    def calculate_dpo_loss(
        self,
        chosen_logits: torch.Tensor,
        chosen_ref_log_probs: torch.Tensor,
        chosen_labels: torch.Tensor,
        chosen_labels_sft: torch.Tensor,
        rejected_logits: torch.Tensor,
        rejected_ref_log_probs: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the DPO loss

        Input:
            chosen_logits (torch.Tensor):
                The logits for the chosen trajectories

                Shape: (batch size, sequence length - 1, vocabulary size)

            chosen_ref_log_probs (torch.Tensor):
                The precomputed reference log probs for chosen trajectories

                Shape: (batch size, sequence length - 1)

            chosen_labels (torch.Tensor):
                The labels for the chosen sequence
                Contains tokens with label IGNORE_TOKEN_ID,
                which we ignore from DPO loss (typically non-assistant tokens)

                Shape: (batch size, sequence length - 1)

            chosen_labels_sft (torch.Tensor):
                The labels for the chosen sequence
                Contains tokens with label IGNORE_TOKEN_ID,
                which we ignore from SFT loss (typically non-assistant tokens)

                NOTE: SFT loss is calculated on end of sequence tokens (e.g., <eot_id>)
                But DPO loss is not

                Shape: (batch size, sequence length - 1)

            rejected_logits (torch.Tensor):
                The logits for the rejected trajectories

                Shape: (batch size, sequence length - 1, vocabulary size)

            rejected_ref_log_probs (torch.Tensor):
                The precomputed reference log probs for rejected trajectories

                Shape: (batch size, sequence length - 1)

            rejected_labels (torch.Tensor):
                The labels for the rejected sequence
                Contains tokens with label IGNORE_TOKEN_ID,
                which we ignore from loss (typically non-assistant tokens)

                Shape: (batch size, sequence length - 1)
        """
        # calculate log probabilities
        chosen_log_probs = get_log_probs(
            logits=chosen_logits,
            labels=chosen_labels,
        )
        chosen_mask = chosen_labels != IGNORE_TOKEN_ID
        chosen_ref_log_probs = (chosen_ref_log_probs * chosen_mask).sum(dim=-1)

        rejected_log_probs = get_log_probs(
            logits=rejected_logits,
            labels=rejected_labels,
        )
        rejected_mask = rejected_labels != IGNORE_TOKEN_ID
        rejected_ref_log_probs = (rejected_ref_log_probs * rejected_mask).sum(dim=-1)

        # calculate log probability ratios
        chosen_log_ratio = chosen_log_probs - chosen_ref_log_probs
        rejected_log_ratio = rejected_log_probs - rejected_ref_log_probs

        dpo_loss = -F.logsigmoid(self.beta * (chosen_log_ratio - rejected_log_ratio))

        # If rpo_alpha > 0, we add an sft loss on chosen trajectories to the DPO loss
        # See this for more details: https://arxiv.org/abs/2404.19733
        if self.rpo_alpha > 0.0:
            nll_loss = -get_log_probs(
                logits=chosen_logits,
                labels=chosen_labels_sft,
                average_log_prob=True,
            )
            dpo_loss = dpo_loss + self.rpo_alpha * nll_loss

        return dpo_loss.mean()
