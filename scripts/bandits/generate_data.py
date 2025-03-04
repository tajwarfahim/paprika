import argparse
import numpy as np
from typing import Dict, Any, List
import os

from llm_exploration.bandits.llm_bandit import LLMBanditExtended, LLMBandit
from llm_exploration.bandits.bandit_algorithms import (
    BanditAlgorithm,
    UCB,
    EpsilonGreedy,
    ThompsonSampling,
)
from llm_exploration.constants import DATA_DIR, PARENT_DIR
from llm_exploration.utils.data_utils import write_json, read_json
from llm_exploration.utils.stats_utils import get_cumulative_average
from llm_exploration.bandits.bandit_utils import get_empirical_regret


def get_arguments() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-num_data",
        "--num_data",
        type=int,
        default=100,
    )

    ap.add_argument(
        "-ucb_alpha",
        "--ucb_alpha",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "-epsilon",
        "--epsilon",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "-savedir",
        "--savedir",
        type=str,
        default=DATA_DIR,
    )

    ap.add_argument(
        "-data_type",
        "--data_type",
        type=str,
        default="train",
        choices=["train", "eval"],
    )

    ap.add_argument(
        "-include_special_tokens",
        "--include_special_tokens",
        action="store_true",
    )

    ap.add_argument(
        "-include_system_prompt",
        "--include_system_prompt",
        action="store_true",
    )

    ap.add_argument(
        "-include_text_explanation",
        "--include_text_explanation",
        action="store_true",
    )

    script_arguments = vars(ap.parse_args())
    return script_arguments


def load_bandit_config_file(args) -> List[Dict[str, Any]]:
    config_file_location = os.path.join(
        PARENT_DIR, "llm_exploration", "bandits", f"{args['data_type']}_bandit_configs.json"
    )
    return read_json(config_file_location)


def get_algorithm(algorithm_name: str) -> BanditAlgorithm:
    name_to_algorithm = {
        "ucb": UCB,
        "epsilon_greedy": EpsilonGreedy,
        "thompson_sampling": ThompsonSampling,
    }

    if algorithm_name not in name_to_algorithm:
        raise ValueError(f"Given algorithm {args['algorithm']} is not supported.")

    return name_to_algorithm[algorithm_name]


def get_algorithm_hyperparams(
    args: Dict[str, Any],
    algorithm_name: str,
    bandit_env: LLMBandit,
) -> Dict[str, Any]:
    if algorithm_name == "ucb":
        return {
            "num_arms": len(bandit_env.get_probability_list()),
            "alpha": args["ucb_alpha"],
        }
    elif algorithm_name == "epsilon_greedy":
        return {
            "num_arms": len(bandit_env.get_probability_list()),
            "epsilon": args["epsilon"],
        }
    elif algorithm_name == "thompson_sampling":
        return {
            "num_arms": len(bandit_env.get_probability_list()),
        }

    else:
        raise ValueError(f"Given algorithm {algorithm_name} is not supported.")


def run_script():
    args = get_arguments()
    bandit_configs = load_bandit_config_file(args=args)

    all_trajectories = []

    for data_generation_turn in range(args["num_data"]):
        bandit_config_index = data_generation_turn % len(bandit_configs)
        bandit_config = bandit_configs[bandit_config_index]
        probability_list = np.random.uniform(
            low=0.0,
            high=1.0,
            size=len(bandit_config["arms"]),
        ).tolist()

        bandit_env = LLMBanditExtended(
            probability_list=probability_list,
            arms_list=bandit_config["arms"],
            T=bandit_config["num_turns"],
            mode="original",
            reward_history_type="original",
            include_system_prompt_in_conversation=args["include_system_prompt"],
            include_special_tokens=args["include_special_tokens"],
            include_text_explanation=args["include_text_explanation"],
            system_prompt=bandit_config["system_prompt"],
            answer=bandit_config["answer"],
        )

        algorithm_name = np.random.choice(["ucb", "thompson_sampling"])
        algorithm_class = get_algorithm(algorithm_name=algorithm_name)
        algorithm_kwargs = get_algorithm_hyperparams(
            algorithm_name=algorithm_name,
            args=args,
            bandit_env=bandit_env,
        )
        algorithm: BanditAlgorithm = algorithm_class(
            **algorithm_kwargs,
        )

        run_hyperparams = {
            "bandit_env": bandit_env,
            "T": bandit_config["num_turns"],
        }

        _ = algorithm.run_algorithm(**run_hyperparams)
        trajectory = bandit_env.generate_history(
            use_text_actions=args["include_text_explanation"],
        )

        datapoint = {
            "conversation": trajectory,
            "all_rewards": bandit_env.get_rewards(),
            "cumulative_average_rewards": get_cumulative_average(arr=bandit_env.get_rewards()),
            "empirical_regret": get_empirical_regret(bandit_env=bandit_env),
            "probability_list": bandit_env.get_probability_list(),
        }
        all_trajectories.append(datapoint)

    data = {
        "num_trials": args["num_data"],
        "records": all_trajectories,
    }

    save_sub_dir = (
        f"llm_evaluation_trajectories_"
        f"{args['data_type']}_"
        f"special_tokens_{args['include_special_tokens']}_"
        f"system_prompt_{args['include_system_prompt']}_"
        f"text_explanation_{args['include_text_explanation']}"
    )

    write_json(
        fname=os.path.join(
            args["savedir"],
            save_sub_dir,
            f"llm_bandit_trajectories.json",
        ),
        data=data,
    )


if __name__ == "__main__":
    run_script()
