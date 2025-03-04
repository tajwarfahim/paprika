# import from general packages
import os
from typing import Tuple
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.trainer_pt_utils import LabelSmoother
import torch
import math
import pathlib

# imports from our packages
from llm_exploration.bandits.bandit_training_arguments import (
    load_bandit_training_arguments,
    DataClass,
)
from llm_exploration.utils.torch_utils import set_seed_everywhere
from llm_exploration.utils.huggingface_utils import safe_save_model_for_hf_trainer
from llm_exploration.common.tokenizer_separators import (
    get_tokenizer_separators,
    TokenizerSeparators,
)
from llm_exploration.bandits.bandit_datasets import get_bandit_dataset
from llm_exploration.llm_finetuning import get_trainer
from llm_exploration.bandits.bandit_trainer_callback import BanditTrainerCallback
from llm_exploration.bandits.llm_bandit import LLMBandit


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
local_rank = 0


def rank0_print(*args):
    """
    Only print the given arg if the current process has rank_0.
    """
    if local_rank == 0:
        print(*args)


def get_datasets(
    training_args: DataClass,
    tokenizer: AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Get train and eval dataset for multiturn sft.

    Input:
        training_args (DataClass):
            The parsed arguments for this training script.

        tokenizer (transformers.AutoTokenizer):
            the tokenizer for the particular model.

        tokenizer_separator (TokenizerSeparators):
            tokenizer separator used for this model.

    Output:
        train_dataset (BanditSFTDataset):
            dataset used for training

        eval_dataset (BanditSFTDataset):
            dataset used for evaluation
    """
    train_dataset = get_bandit_dataset(
        trainer_type=training_args.trainer_type,
        data_dir=training_args.train_data_dir,
        tokenizer=tokenizer,
        tokenizer_separator=tokenizer_separator,
        ignore_token_id=IGNORE_TOKEN_ID,
        num_samples=training_args.num_train_samples,
        rwr_gamma=training_args.rwr_gamma,
        rwr_min_reward=training_args.rwr_min_reward,
    )

    eval_dataset = get_bandit_dataset(
        trainer_type=training_args.trainer_type,
        data_dir=training_args.train_data_dir,
        tokenizer=tokenizer,
        tokenizer_separator=tokenizer_separator,
        ignore_token_id=IGNORE_TOKEN_ID,
        num_samples=training_args.num_eval_samples,
        rwr_gamma=training_args.rwr_gamma,
        rwr_min_reward=training_args.rwr_min_reward,
    )

    return train_dataset, eval_dataset


def load_model_and_tokenizer(
    training_args: DataClass,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, TokenizerSeparators,]:
    """
    Loads the model, tokenizer and tokenizer_separator.

    Input:
        training_args (DataClass):
            The arguments (parsed + default) used for training.

    Output:
        tokenizer (transformers.AutoTokenizer):
            tokenizer for the particular model.

        model (transformers.AutoModelForCausalLM):
            pretrained weights for the LLM that will be used.

        tokenizer_separator (TokenizerSeparators):
            tokenizer separator for this particular model.
    """
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=training_args.model_name_or_path,
        trust_remote_code=True,
    )

    # Set RoPE scaling factor - useful when you want to train with a larger context window
    if training_args.use_rope:
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {
                "type": "linear",
                "factor": scaling_factor,
            }

    config.use_cache = False

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=training_args.model_name_or_path,
        trust_remote_code=True,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.tie_weights()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=training_args.model_name_or_path,
        trust_remote_code=True,
        config=config,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # Add padding token to the tokenizer + Add masking values.
    # tokenizer separator will be used for masking in dataloader
    tokenizer, tokenizer_separator = get_tokenizer_separators(
        tokenizer=tokenizer,
        tokenizer_name=training_args.tokenizer_name,
    )

    print("\nAdding <Answer>, </Answer>tokens to tokenizer", training_args.add_answer_tokens)
    if training_args.add_answer_tokens:
        print("\nAdding tokens to tokenizer.\n")
        tokenizer.add_tokens(["<Answer>"], special_tokens=False)
        tokenizer.add_tokens(["</Answer>"], special_tokens=False)

    # NOTE: if the token_id exceed the vocab_size will cause failing in training process!
    # we need to add special config and resize the embedding size!
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model, tokenizer_separator


def train():
    """
    Main entry point for running this script.
    Trains a model on trajectories for the bandit problem.
    """
    global local_rank
    # Parse args; All configs sent to the model
    training_args = load_bandit_training_arguments()
    os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # Seed everything
    set_seed_everywhere(training_args.seed)

    local_rank = training_args.local_rank

    tokenizer, model, tokenizer_separator = load_model_and_tokenizer(
        training_args=training_args,
    )

    train_dataset, eval_dataset = get_datasets(
        training_args=training_args,
        tokenizer=tokenizer,
        tokenizer_separator=tokenizer_separator,
    )

    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    if training_args.generate_during_training:
        bandit_env = LLMBandit(
            arms_list=["blue", "red", "green"],
            probability_list=[0.7, 0.3, 0.3],
            T=50,
            mode="original",
            reward_history_type="original",
            name="easy",
            include_special_tokens=True,
            include_text_explanation=True,
        )

        bandit_callback = BanditTrainerCallback(
            tokenizer=tokenizer,
            bandit_env=bandit_env,
            n_steps=training_args.eval_steps,
            model_name=training_args.model_name_or_path,
            savedir=training_args.eval_samples_savedir,
            num_trials=10,
        )
        trainer.add_callback(bandit_callback)

    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and training_args.should_resume_training_from_checkpoint
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
