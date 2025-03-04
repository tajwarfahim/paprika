from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from dataclasses import dataclass, field
from typing import Optional, NewType, Any


DataClass = NewType("DataClass", Any)


@dataclass
class BanditTrainingArguments(TrainingArguments):
    """
    Training arguments for the Bandit problem.
    """

    cache_dir: Optional[str] = field(default="/data/user_data/ftajwar/training_cache")
    output_dir: str = field(default="/data/user_data/ftajwar/training_outputs")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    torch_empty_cache_steps: int = field(
        default=1, metadata={"help": "Number of steps to call torch.cuda.empty_cache()"}
    )

    train_data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    train_dataset_path_dpo: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to training data with precomputed ref log probs, used for DPO."
        },
    )
    eval_dataset_path_dpo: Optional[str] = field(
        default=None,
        metadata={"help": "Path to eval data with precomputed ref log probs, used for DPO"},
    )

    eval_samples_savedir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save eval samples during training."},
    )
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    wandb_entity: str = field(
        default="llm_exploration",
        metadata={"help": "The entity for wandb logging."},
    )

    wandb_project: str = field(
        default="bandit",
        metadata={"help": "The project under which runs should be saved."},
    )

    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention or not."},
    )

    num_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of datapoints to train on."},
    )

    num_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of datapoints on which we should run evaluation."},
    )

    tokenizer_name: str = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={
            "help": "Name of the tokenizer base model, used for obtaining tokenizer separator."
        },
    )

    add_answer_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to add <Answer>, </Answer> tokens to the tokenizer."},
    )

    use_rope: bool = field(
        default=False,
        metadata={"help": "Whether to use RoPE scaling for long context window."},
    )

    trainer_type: str = field(
        default="SFT",
        metadata={"help": "What type of trainer to use."},
    )

    gem_beta: float = field(
        default=0.7,
        metadata={"help": "The beta parameter for GEM loss."},
    )

    gem_h_function: str = field(
        default="linear",
        metadata={"help": "The h function for GEM loss."},
    )

    rwr_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for RWR loss."},
    )

    rwr_type: str = field(
        default="exp",
        metadata={"help": "The reward weight type for RWR loss."},
    )

    rwr_term_min: float = field(
        default=0.0,
        metadata={"help": "The minimum value that we clip the RWR term to."},
    )

    rwr_term_max: float = field(
        default=3.0, metadata={"help": "The maximum value that we clip the RWR term to."}
    )

    rwr_gamma: float = field(
        default=1.0,
        metadata={"help": "The reward discount factor, gamma, per turn."},
    )

    rwr_min_reward: float = field(
        default=0.0,
        metadata={"help": "Min reward per turn for RWR."},
    )

    generate_during_training: bool = field(
        default=False,
        metadata={"help": "Whether to eval on real bandit trajectories during training."},
    )

    dpo_beta: float = field(
        default=0.1,
        metadata={
            "help": (
                "The beta parameter for running DPO: "
                "higher beta means stay closer to reference policy."
            )
        },
    )

    rejected_trajectory_sampling_strategy: str = field(
        default="worst",
        metadata={"help": "Strategy to pick the rejected trajectory."},
    )

    simpo_beta: float = field(
        default=2.5,
        metadata={"help": "Beta parameter for SimPO loss."},
    )

    simpo_gamma: float = field(
        default=1.4,
        metadata={"help": "Gamma parameter for SimPO gamma."},
    )

    torch_empty_cache_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to call torch.cuda.empty_cache()"},
    )

    rpo_alpha: float = field(
        default=0.0,
        metadata={"help": "Alpha parameter in RPO (Iter. Res. PO) loss."},
    )

    should_resume_training_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to resume training from checkpoint states."},
    )

    num_turn_threshold_for_dpo: int = field(
        default=4,
        metadata={
            "help": "Minimum difference in num_turns for chosen and rejected trajectory."
        },
    )

    judge_label_strategy: str = field(
        default="count_invalids_as_failures",
        metadata={"help": "How to deal with judge labels regarding invalid trajectories."},
    )

    token_length_threshold: int = field(
        default=2500,
        metadata={"help": "Upper bound on number of tokens on filtered trajectories."},
    )

    regularization_coefficient: float = field(
        default=1.0,
        metadata={"help": "Coefficient on the reward sum penalty for DPO"},
    )

    use_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether to use Liger Kernel for training."},
    )

    copy_instruct_tokenizer_chat_template: bool = field(
        default=True,
        metadata={"help": "Whether to copy the chat template from the instruct version."},
    )


def load_bandit_training_arguments() -> DataClass:
    """
    Returns the parsed arguments for running training
    on the bandit problem.

    Input:
        None

    Output:
        training_args (DataClass):
            A dataclass object, containing the arguments for training
            the model.
    """
    parser = HfArgumentParser(BanditTrainingArguments)
    return parser.parse_args_into_dataclasses()[0]
