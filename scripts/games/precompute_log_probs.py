# Import from general libraries
import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Set
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from collections import defaultdict
from transformers.trainer_pt_utils import LabelSmoother

# Import from our own packages
from llm_exploration.constants import (
    PARENT_DIR,
    DATAGEN_CONFIG_DIR,
)
from llm_exploration.utils.torch_utils import set_seed_everywhere
from llm_exploration.inference import HuggingFaceLLMInferenceEngine
from llm_exploration.common.tokenizer_separators import get_tokenizer_separators
from llm_exploration.game.dataset import GameDPODataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def load_inference_engine(config: Dict[str, Any]) -> HuggingFaceLLMInferenceEngine:
    """
    Given config, loads the appropriate inference engine.

    Input:
        config (Dict):
            configuration dictionary

    Output:
        inference_engine (HuggingFaceLLMInferenceEngine):
            the inference engine to use
    """
    if not torch.cuda.is_available():
        raise ValueError(f"This script cannot be run without GPUs, could not find one.")

    model_name = config["inference_engine"]["model_name"]
    model_type = config["inference_engine"]["model_type"]
    tokenizer_name = config["inference_engine"]["tokenizer_name"]

    if model_type != "huggingface_models":
        raise ValueError(f"Given model type {model_type} not supported.")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name,
        trust_remote_code=True,
        use_fast=False,
        model_max_length=config["inference_engine"]["model_max_length"],
        padding_side="left",
    )

    tokenizer, tokenizer_seperator = get_tokenizer_separators(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
    )

    model.resize_token_embeddings(len(tokenizer))

    inference_engine = HuggingFaceLLMInferenceEngine(
        model_name=config["inference_engine"]["base_model_name"],
        model=model,
        tokenizer=tokenizer,
    )

    return inference_engine, tokenizer_seperator


def precompute_logits(
    dataset: GameDPODataset,
    inference_engine: HuggingFaceLLMInferenceEngine,
) -> defaultdict:
    """
    Precomputes the log probabilities for inputs in a
    DPO dataset, and returns them.

    Input:
        dataset (GameDPODataset):
            The DPO dataset for which we want to precompute
            log probabilities

        inference_engine (HuggingFaceLLMInferenceEngine):
            Inference engine to use, to precompute
            log probabilities

    Output:
        outputs (defaultdict):
            Should have the following format:
            {
                "chosen_input_ids": input_id (tensor),
                "chosen_labels": label (tensor),
                "chosen_attention_mask": attention_mask (tensor),
                "chosen_log_probs": chosen_log_probs (tensor),
                "rejected_input_ids": input_id (tensor),
                "rejected_labels": label (tensor),
                "rejected_attention_mask": attention_mask (tensor),
                "rejected_log_probs": rejected_log_probs (tensor),
            }
    """
    output = defaultdict(list)

    for idx in tqdm(range(len(dataset))):
        datapoint = dataset[idx]

        datapoint["chosen_log_probs"] = inference_engine.compute_log_probs(
            input_ids=datapoint["chosen_input_ids"].unsqueeze(dim=0),
            labels=datapoint["chosen_labels"].unsqueeze(dim=0),
            attention_mask=datapoint["chosen_attention_mask"].unsqueeze(dim=0),
        ).squeeze()

        datapoint["rejected_log_probs"] = inference_engine.compute_log_probs(
            input_ids=datapoint["rejected_input_ids"].unsqueeze(dim=0),
            labels=datapoint["rejected_labels"].unsqueeze(dim=0),
            attention_mask=datapoint["rejected_attention_mask"].unsqueeze(dim=0),
        ).squeeze()

        for key in datapoint:
            output[key].append(datapoint[key])

    for key in output:
        output[key] = torch.stack(output[key], dim=0)

    return output


@hydra.main(
    version_base=None,
    config_path=DATAGEN_CONFIG_DIR,
    config_name="precompute_log_probs",
)
def main(config: DictConfig):
    """
    Main entry point for precomputing the logits, given a
    DPO dataset and a reference model.
    """
    config.repo_dir = PARENT_DIR
    OmegaConf.resolve(config)
    set_seed_everywhere(config.seed)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    config = OmegaConf.to_object(config)
    print(config)

    inference_engine, tokenizer_separator = load_inference_engine(config=config)

    dataset = GameDPODataset(
        data_dir=config["data_dir"],
        tokenizer=inference_engine.tokenizer,
        tokenizer_separator=tokenizer_separator,
        ignore_token_id=IGNORE_TOKEN_ID,
        rejected_trajectory_sampling_strategy=config["rejected_sampling"],
        threshold=config["num_turn_threshold_for_dpo"],
        judge_label_strategy=config["judge_label_strategy"],
        num_samples=config["num_samples"],
        token_length_threshold=config["token_length_threshold"],
    )

    if config["should_save_trajectories_to_json"]:
        dataset.save_dataset_trajectories_to_json(config["save_trajectories_file"])

    output = precompute_logits(
        dataset=dataset,
        inference_engine=inference_engine,
    )

    torch.save(output, config["save_file"])


if __name__ == "__main__":
    main()
