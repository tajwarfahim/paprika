from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from torch.utils.data import Dataset

from llm_exploration.llm_finetuning.gem_trainer import GEMTrainer
from llm_exploration.llm_finetuning.rwr_trainer import OfflineRWRTrainer
from llm_exploration.llm_finetuning.dpo_trainer import DPOTrainer
from llm_exploration.llm_finetuning.simpo_trainer import SimPOTrainer
from llm_exploration.llm_finetuning.length_controlled_dpo_trainer import (
    LengthControlledDPOTrainer,
)
from llm_exploration.llm_finetuning.regularized_dpo_trainer import RegularizedDPOTrainer
from llm_exploration.bandits.bandit_training_arguments import DataClass


def get_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: DataClass,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    """
    Helper function to load different types of model trainers.

    Input:
        model (AutoModelForCausalLM):
            The model that will be trained

        tokenizer (AutoTokenizer):
            tokenizer for this particular model/training run

        args (DataClass):
            arguments for the training script, should contain the necessary
            hyperparameters.

        train_dataset (Dataset):
            Training dataset for this trainer

        eval_dataset (Dataset):
            Eval/validation dataset for this trainer

    Output:
        trainer (Trainer):
            An instance of Huggingface Trainer or one if its inherited classes.
    """
    model_loading_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    if args.trainer_type == "SFT":
        trainer_class = Trainer

    elif args.trainer_type == "GEM":
        trainer_class = GEMTrainer
        model_loading_kwargs["beta"] = args.gem_beta
        model_loading_kwargs["h_function"] = args.gem_h_function

    elif args.trainer_type == "OfflineRWR":
        trainer_class = OfflineRWRTrainer
        model_loading_kwargs["rwr_temperature"] = args.rwr_temperature
        model_loading_kwargs["rwr_type"] = args.rwr_type
        model_loading_kwargs["rwr_term_max"] = args.rwr_term_max
        model_loading_kwargs["rwr_term_min"] = args.rwr_term_min

    elif args.trainer_type == "DPO":
        trainer_class = DPOTrainer
        model_loading_kwargs["beta"] = args.dpo_beta
        model_loading_kwargs["rpo_alpha"] = args.rpo_alpha

    elif args.trainer_type == "SimPO":
        trainer_class = SimPOTrainer
        model_loading_kwargs["beta"] = args.simpo_beta
        model_loading_kwargs["simpo_gamma"] = args.simpo_gamma

    elif args.trainer_type == "LengthControlledDPO":
        trainer_class = LengthControlledDPOTrainer
        model_loading_kwargs["beta"] = args.dpo_beta
        model_loading_kwargs["rpo_alpha"] = args.rpo_alpha

    elif args.trainer_type == "RegularizedDPO":
        trainer_class = RegularizedDPOTrainer
        model_loading_kwargs["beta"] = args.dpo_beta
        model_loading_kwargs["regularization_coefficient"] = args.regularization_coefficient

    else:
        raise ValueError(f"Given trainer type {args.trainer_type} is not supported.")

    return trainer_class(**model_loading_kwargs)
