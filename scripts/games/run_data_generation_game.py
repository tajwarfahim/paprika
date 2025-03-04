# Import from general libraries
import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Set, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# Import from our own packages
from llm_exploration.constants import (
    PARENT_DIR,
    DATAGEN_CONFIG_DIR,
)
from llm_exploration.utils.data_utils import (
    write_json,
    read_json,
)
from llm_exploration.utils.temperature_annealing_utils import get_min_p_from_temperature
from llm_exploration.utils.torch_utils import set_seed_everywhere
import llm_exploration.inference as InferenceEngine
from llm_exploration.common.tokenizer_separators import get_tokenizer_separators
from llm_exploration.game import (
    get_game_environment,
    GameSimulator,
)


def load_inference_engine(
    config: Dict[str, Any], num_gpus_being_used: int
) -> Tuple[InferenceEngine.LLMInferenceEngine, int]:
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

    elif model_type == "vllm_models":
        inference_engine = InferenceEngine.VLLMInferenceEngine(
            model_name=model_name,
            max_model_len=config["model_max_length"],
        )

    elif model_type == "huggingface_models":
        device_name = f"cuda:{num_gpus_being_used}"
        num_gpus_being_used += 1

        print("\nLoading model: ", model_name, "\n")

        model_loading_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        if config["use_flash_attention"]:
            model_loading_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(**model_loading_kwargs).to(device_name)

        model.config.use_cache = False
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

    elif model_type == "wordle_inference_engine":
        inference_engine = InferenceEngine.WordleInferenceEngine(
            mode=config["mode"],
        )

    elif model_type == "wordle_modified_inference_engine":
        inference_engine = InferenceEngine.WordleModifiedInferenceEngine(
            mode=config["mode"],
        )

    elif model_type == "cellular_automata_inference_engine":
        inference_engine = InferenceEngine.CellularAutomationInferenceEngine(
            mode=config["mode"],
        )

    elif model_type == "jericho_inference_engine":
        inference_engine = InferenceEngine.JerichoInferenceEngine(
            env_name=config["env_name"],
        )

    elif model_type == "bandit_inference_engine":
        inference_engine = InferenceEngine.BanditInferenceEngine(
            difficulty=config["difficulty"],
            randomize_arm_probabilities=config["randomize_arm_probabilities"],
        )

    elif model_type == "bandit_bai_fixed_budget_inference_engine":
        inference_engine = InferenceEngine.BanditBAIFixedBudgetInferenceEngine(
            difficulty=config["difficulty"],
            randomize_arm_probabilities=config["randomize_arm_probabilities"],
        )

    elif model_type == "minesweeper_inference_engine":
        inference_engine = InferenceEngine.MinesweeperInferenceEngine()

    elif model_type == "minesweeper_judge_inference_engine":
        inference_engine = InferenceEngine.MinesweeperJudgeInferenceEngine()

    elif model_type == "mastermind_inference_engine":
        inference_engine = InferenceEngine.MastermindInferenceEngine(
            mode=config["mode"],
        )

    elif model_type == "battleship_inference_engine":
        inference_engine = InferenceEngine.BattleshipInferenceEngine()

    else:
        raise ValueError(f"Given model_type {model_type} is not supported.")

    return inference_engine, num_gpus_being_used


def check_available_gpus(config: Dict[str, Any]) -> None:
    """
    Checks if the job has available number of GPUS.

    Input:
        config (Dict[str, Any]):
            config for agent and environment

    Output:
        None
    """
    num_gpus_required = 0
    if config["agent"]["model_type"] == "huggingface_models":
        num_gpus_required += 1
    if config["env"]["model_type"] == "huggingface_models":
        num_gpus_required += 1

    if num_gpus_required > 0:
        if not torch.cuda.is_available():
            raise ValueError(
                f"No GPUs available, while we needed {num_gpus_required} to run the job."
            )

        num_gpus_available = torch.cuda.device_count()

        if num_gpus_available < num_gpus_required:
            raise ValueError(
                f"Not enough GPUs: required: {num_gpus_required}, "
                f"available: {num_gpus_available}"
            )


@hydra.main(
    version_base=None,
    config_path=DATAGEN_CONFIG_DIR,
    config_name="run_data_generation_game",
)
def main(config: DictConfig):
    """
    Main entry point for running game for data generation.
    """
    config.repo_dir = PARENT_DIR
    OmegaConf.resolve(config)
    set_seed_everywhere(config.seed)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    config = OmegaConf.to_object(config)
    print(config)

    check_available_gpus(config=config)

    num_gpus_being_used = 0
    agent, num_gpus_being_used = load_inference_engine(
        config=config["agent"],
        num_gpus_being_used=num_gpus_being_used,
    )
    env, num_gpus_being_used = load_inference_engine(
        config=config["env"],
        num_gpus_being_used=num_gpus_being_used,
    )
    judge, num_gpus_being_used = load_inference_engine(
        config=config["judge"],
        num_gpus_being_used=num_gpus_being_used,
    )
    print(f"\nUsing {num_gpus_being_used} GPUs.\n")

    game = GameSimulator(agent=agent, env=env, judge=judge)

    game_environment = get_game_environment(
        environment_name=config["game_env"]["environment_name"]
    )
    game_scenarios = game_environment.get_game_scenarios(config=config["game_env"])

    # pick all indices if there is no given curriculum
    data_indices = [index for index in range(len(game_scenarios))]

    if config["curriculum_file"] is not None:
        data_indices = read_json(config["curriculum_file"])["batch_indices"]

    save_file = config["save_file"]
    if config["curriculum"] is not None:
        save_file_parts = save_file.split("/")
        save_file_name = save_file_parts[-1]
        save_file_parts[-1] = f"curriculum_{config['curriculum']}"
        save_file_parts.append(
            f"curriculum_round_{config['curriculum_round']}_{save_file_name}"
        )
        save_file = "/".join(save_file_parts)

    print("\nWill save output to: ", save_file, "\n")

    all_histories = []

    for i in range(config["start_index"], config["end_index"]):
        scenario_history = []
        game_scenario_index = data_indices[i]

        print("\nScenario: ", i - config["start_index"] + 1)
        print("Game scenario index: ", game_scenario_index, "\n")

        game.reset()

        for _ in range(config["num_trajectories_per_game_scenario"]):
            # Prepare game scenario
            situation_config = {
                "env_input": game_scenarios[game_scenario_index]["env"],
                "agent_input": game_scenarios[game_scenario_index]["agent"],
            }

            env_first_message = game_environment.get_env_message(
                config=situation_config,
            )
            agent_first_message = game_environment.get_agent_message(config=situation_config)

            env_optional_message = game_environment.get_env_optional_message(
                config=situation_config,
            )
            agent_optional_message = game_environment.get_agent_optional_message(
                config=situation_config,
            )

            env_response_extractor = game_environment.get_enviroment_response_extractor()
            agent_response_extractor = game_environment.get_agent_response_extractor()
            verifier_input_generator = game_environment.get_verifier_input_generator()

            # Run the game through the simulator
            record = game.run_one_iteration(
                agent_game_scenario=game_scenarios[game_scenario_index]["agent"],
                env_game_scenario=game_scenarios[game_scenario_index]["env"],
                env_first_message=env_first_message,
                agent_first_message=agent_first_message,
                max_turns=game_environment.get_game_max_turns(),
                agent_temperature=config["agent_temperature"],
                agent_top_p=config["agent_top_p"],
                agent_min_p=get_min_p_from_temperature(
                    temperature=config["agent_temperature"],
                    temperature_threshold=config["temperature_threshold"],
                    min_p_choice=config["min_p_choice"],
                ),
                agent_max_n_tokens=config["agent_max_n_tokens"],
                env_temperature=config["env_temperature"],
                env_top_p=config["env_top_p"],
                env_min_p=get_min_p_from_temperature(
                    temperature=config["env_temperature"],
                    temperature_threshold=config["temperature_threshold"],
                    min_p_choice=config["min_p_choice"],
                ),
                env_max_n_tokens=config["env_max_n_tokens"],
                judge_max_n_tokens=config["judge_max_n_tokens"],
                judge_temperature=config["judge_temperature"],
                judge_top_p=config["judge_top_p"],
                judge_min_p=get_min_p_from_temperature(
                    temperature=config["judge_temperature"],
                    temperature_threshold=config["temperature_threshold"],
                    min_p_choice=config["min_p_choice"],
                ),
                terminate_at_first_agent_failure=config["terminate_at_first_agent_failure"],
                env_optional_message=env_optional_message,
                agent_optional_message=agent_optional_message,
                env_response_extractor=env_response_extractor,
                agent_response_extractor=agent_response_extractor,
                num_max_env_response_generations=config["num_max_env_response_generations"],
                num_max_agent_response_generations=config[
                    "num_max_agent_response_generations"
                ],
                env_default_response=game_environment.get_environment_default_response(),
                judge_prompt_env=game_environment.get_judge_prompt_env(
                    config=situation_config,
                ),
                judge_prompt_agent=game_environment.get_judge_prompt_agent(
                    config=situation_config,
                ),
                verifier_input_generator=verifier_input_generator,
                agent_model_supports_system_message=(
                    config["agent_model_supports_system_message"]
                ),
            )

            # checking if the game trajectory is valid
            if record is not None:
                scenario_history.append(record)

        # Only record it if we have enough successful responses
        if len(scenario_history) > 0:
            all_histories.append(scenario_history)

    print("\nSaving output to: ", save_file, "\n")
    write_json(
        data={
            "records": all_histories,
        },
        fname=save_file,
    )


if __name__ == "__main__":
    main()
