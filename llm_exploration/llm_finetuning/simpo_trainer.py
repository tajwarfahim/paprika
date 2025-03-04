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


class SimPOTrainer(Trainer):
    """
    A minimal implementation of the SimPO algorithm, described in this paper:
    SimPO: Simple Preference Optimization with a Reference-Free Reward
    (https://arxiv.org/abs/2405.14734)

    This algorithm is a variation of DPO (https://arxiv.org/abs/2305.18290):
        1. without the need of reference model
           (in case the reference model is bad, this is helpful,
           but also has risks of reward over-optimization)
        2. averages log probabilities per token instead of summing them,
           (reducing length exploitation)
        3. Imposes a margin between the reward of preferred and
           dispreferred responses.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of this trainer class.

        Special inputs besides the regular one that the parent class takes:
            beta (float):
                The beta parameter in the SimPO loss
                Default: 0.1

            simpo_gamma (float):
                Margin between preferred and dispreferred responses'
                reward
        """
        self.beta = kwargs.pop("beta", 2.5)
        self.simpo_gamma = kwargs.pop("simpo_gamma", 1.4)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tuple[Any, torch.Tensor], torch.Tensor]:
        """
        Computes and returns loss, defined in the SimPO paper.
        Input:
            model (PreTrainedModel or torch.nn.Module):
                A language model that we are trying to finetune.

            inputs (Dict[str, torch.Tensor]):
                The inputs to the language model. This needs to have
                the following shape:

                {
                    "input_ids": chosen_input_id (tensor),
                    "labels": chosen_label (tensor),
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
            "labels": inputs["labels"],
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
            "labels": inputs["rejected_labels"],
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

        if shift_labels:
            chosen_labels = chosen_labels[:, 1:]
            rejected_labels = rejected_labels[:, 1:]

            chosen_logits = chosen_logits[:, :-1, :]
            rejected_logits = rejected_logits[:, :-1, :]

        loss = self.calculate_simpo_loss(
            chosen_logits=chosen_logits,
            chosen_labels=chosen_labels,
            rejected_logits=rejected_logits,
            rejected_labels=rejected_labels,
        )

        if return_outputs:
            return (loss, chosen_model_outputs)
        else:
            return loss

    def calculate_simpo_loss(
        self,
        chosen_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_logits: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the SimPO loss

        Input:
            chosen_logits (torch.Tensor):
                The logits for the chosen trajectories

                Shape: (batch size, sequence length - 1, vocabulary size)

            chosen_labels (torch.Tensor):
                The labels for the chosen sequence
                Contains tokens with label IGNORE_TOKEN_ID,
                which we ignore from loss (typically non-assistant tokens)

                Shape: (batch size, sequence length - 1)

            rejected_logits (torch.Tensor):
                The logits for the rejected trajectories

                Shape: (batch size, sequence length - 1, vocabulary size)

            rejected_labels (torch.Tensor):
                The labels for the rejected sequence
                Contains tokens with label IGNORE_TOKEN_ID,
                which we ignore from loss (typically non-assistant tokens)

                Shape: (batch size, sequence length - 1)
        """
        # calculate log probabilities
        avg_chosen_log_probs = get_log_probs(
            logits=chosen_logits,
            labels=chosen_labels,
            average_log_prob=True,
        )

        avg_rejected_log_probs = get_log_probs(
            logits=rejected_logits,
            labels=rejected_labels,
            average_log_prob=True,
        )

        simpo_logit = (
            self.beta * (avg_chosen_log_probs - avg_rejected_log_probs) - self.simpo_gamma
        )
        simpo_loss = -F.logsigmoid(simpo_logit)

        return simpo_loss.mean()
