# Import from general libraries
import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Set
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from fastchat.model import get_conversation_template

# Import from our own packages
from llm_exploration.constants import (
    PARENT_DIR,
    DATAGEN_CONFIG_DIR,
)
from llm_exploration.utils.data_utils import write_json
from llm_exploration.utils.torch_utils import set_seed_everywhere
import llm_exploration.inference as InferenceEngine
from llm_exploration.common.tokenizer_separators import get_tokenizer_separators
from llm_exploration.game import (
    get_game_environment,
)


def load_inference_engine(
    config: Dict[str, Any], num_gpus_being_used: int
) -> InferenceEngine.LLMInferenceEngine:
    """
    Given config, loads the appropriate inference engine.

    Input:
        config (Dict):
            configuration dictionary

        num_gpus_being_used (int):
            Number of GPUs currently used to load models
            BEFORE loading the intended inference engine.

    Output:
        inference_engine (InferenceEngine.LLMInferenceEngine):
            the inference engine to use

        num_gpus_being_used (int):
            Number of GPUs currently in use,
            AFTER loading the intended inference engine.
    """
    model_name = config["model_name"]
    model_type = config["model_type"]
    tokenizer_name = config["tokenizer_name"]

    if model_type == "openai_api_models":
        inference_engine = InferenceEngine.OpenAIInferenceEngine(
            model_name=model_name,
        )

    elif model_type == "huggingface_models":
        device_name = f"cuda:{num_gpus_being_used}"
        num_gpus_being_used += 1

        print("\nLoading model: ", model_name, "\n")

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device_name)
        model.tie_weights()

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name,
            trust_remote_code=True,
            use_fast=False,
        )

        if not config["finetuned_tokenizer"]:
            tokenizer, _ = get_tokenizer_separators(
                tokenizer=tokenizer,
                tokenizer_name=config["base_model_name"],
            )
            model.resize_token_embeddings(len(tokenizer))

        inference_engine = InferenceEngine.HuggingFaceLLMInferenceEngine(
            model_name=config["base_model_name"],
            model=model,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Given model_type {model_type} is not supported.")

    return inference_engine, num_gpus_being_used


@hydra.main(
    version_base=None,
    config_path=DATAGEN_CONFIG_DIR,
    config_name="generate_twenty_questions_difficulty",
)
def main(config: DictConfig):
    """
    Main entry point for estimating difficulty level for twenty questions
    from an LLM Judge, eg., gpt-4o-mini
    """
    config.repo_dir = PARENT_DIR
    config.save_dir = PARENT_DIR
    OmegaConf.resolve(config)
    set_seed_everywhere(config.seed)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    config = OmegaConf.to_object(config)
    print(config)

    num_gpus_being_used = 0
    judge, num_gpus_being_used = load_inference_engine(
        config=config["judge"],
        num_gpus_being_used=num_gpus_being_used,
    )
    print(f"\nUsing {num_gpus_being_used} GPUs.\n")

    game_environment = get_game_environment(
        environment_name=config["game_env"]["environment_name"]
    )
    game_scenarios = game_environment.get_game_scenarios(config=config["game_env"])

    print("Number of game scenarios: ", len(game_scenarios))

    all_difficulties = {}
    response_map = {
        "<EASY>": "easy",
        "<MEDIUM>": "medium",
        "<HARD>": "hard",
    }

    for i in range(len(game_scenarios)):
        print("\nScenario: ", i, "\n")

        conv = get_conversation_template("gpt-4")

        system_prompt = (
            "You are an expert judge of the game of 20 questions. "
            "I will give you a topic, and you must classify it into easy, medium or hard, "
            "based on an estimate of how easy it is to guess the topic, "
            "and an estimate of how many turns it will take to guess the topic. "
            "Respond in <EASY>, <MEDIUM> or <HARD>."
        )

        conv.set_system_message(system_prompt)

        topic = game_scenarios[i]["env"]
        user_message = f"Your topic is: {topic}"
        conv.append_message(role="user", message=user_message)

        got_valid_response = False

        while not got_valid_response:
            judge_response = judge.generate(
                conv=conv.to_openai_api_messages(),
                temperature=config["judge_temperature"],
                max_n_tokens=config["judge_max_n_tokens"],
                top_p=config["judge_top_p"],
            )
            got_valid_response = judge_response in response_map

        all_difficulties[topic] = response_map[judge_response]

    write_json(
        data=all_difficulties,
        fname=config["save_file"],
    )


if __name__ == "__main__":
    main()
