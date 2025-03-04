from typing import List, Dict, Tuple, Any
import os
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm

from llm_exploration.utils.data_utils import read_json
from llm_exploration.constants import DATA_DIR


def load_tokenizers(model_names: List[str]) -> Dict[str, AutoTokenizer]:
    """
    Given a list of model names, loads the corresponding tokenizers
    and returns a dictionary of them.

    Input:
        model_names (List[str]):
            List of model names for which we want to load tokenizers.

    Output:
        tokenizers (Dict[str, AutoTokenizer]):
            A dictionary of model_name (str) ---> tokenizer (AutoTokenizer)
    """
    tokenizers = {}

    for model_name in model_names:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            use_fast=False,
            trust_remote_code=True,
        )

    return tokenizers


def get_conversation_lengths(
    json_file_path: str,
    tokenizers: Dict[str, AutoTokenizer],
) -> Tuple[Dict[str, List[int]], Dict[bool, List[int]]]:
    """
    Given a json file of conversations, finds the maximum conversation length

    Input:
        json_file_path (str):
            Name of json file where the conversations are stored.

        tokenizers (Dict[str, AutoTokenizer]):
            dictionary with model_name (str) ---> tokenizer (AutoTokenizer)
            mapping.

    Output:
        lengths_map (Dict[str, List[int]]):
            dictionary with model_name (str) ---> length_of_conversations (List[int])
            mapping

        num_turns_map (Dict[bool, List[int]]):
            Number of turns required to solve the task, based on whether the task
            is solved or not.
    """
    assert os.path.isfile(json_file_path) and json_file_path.endswith(".json")
    data = read_json(fname=json_file_path)

    lengths_map = defaultdict(list)
    num_turns_map = defaultdict(list)

    for record_index in tqdm(range(len(data["records"]))):
        record = data["records"][record_index]
        for trial in record:
            if trial["goal_reached"]:
                conversation = trial["conversation"]
                num_turns_map[trial["goal_reached"]].append(trial["num_turns"])

                for model_name in tokenizers:
                    tokenizer = tokenizers[model_name]
                    tokenized_seq = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                    )
                    lengths_map[model_name].append(len(tokenized_seq))

    return lengths_map, num_turns_map


def get_conversation_lengths_for_all_files(
    dir: str,
    tokenizers: Dict[str, AutoTokenizer],
) -> Tuple[Dict[str, List[int]], Dict[bool, List[int]]]:
    """
    Given a directory that contains json files with conversations,
    returns the length of tokenized conversations in all files.

    Input:
        dir (str):
            path to the directory where the files are stored

        tokenizers (Dict[str, AutoTokenizer]):
            dictionary with model_name (str) ---> tokenizer (AutoTokenizer)
            mapping.

    Output:
        lengths_map (Dict[str, List[int]]):
            dictionary with model_name (str) ---> length_of_conversations (List[int])
            mapping

        num_turns_map (Dict[bool, List[int]]):
            Number of turns required to solve the task, based on whether the task
            is solved or not.
    """
    assert os.path.isdir(dir)

    lengths_map = defaultdict(list)
    num_turns_map = defaultdict(list)

    all_files = os.listdir(dir)
    for file_index in tqdm(range(len(all_files))):
        file = all_files[file_index]
        json_file_path = os.path.join(dir, file)
        if os.path.isfile(json_file_path) and json_file_path.endswith(".json"):
            file_length_map, file_num_turns_map = get_conversation_lengths(
                json_file_path=json_file_path,
                tokenizers=tokenizers,
            )

            for key in file_length_map:
                lengths_map[key].extend(file_length_map[key])

            for key in file_num_turns_map:
                num_turns_map[key].extend(file_num_turns_map[key])

    return lengths_map, num_turns_map


def get_arguments() -> Dict[str, Any]:
    """
    Returns the arguments for this script.

    Input:
        None

    Output:
        arguments (Dict[str, Any]):
            The arguments for this script.
    """
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-dir",
        "--dir",
        type=str,
        default=DATA_DIR,
    )

    ap.add_argument(
        "-num_turns_plot_save_path",
        "--num_turns_plot_save_path",
        type=str,
        default=os.path.join(DATA_DIR, "num_turns_plots", "game_plot.png"),
    )

    ap.add_argument(
        "-num_tokens_plot_save_path",
        "--num_tokens_plot_save_path",
        type=str,
        default=os.path.join(DATA_DIR, "num_tokens_plots", "game_plot.png"),
    )

    script_arguments = vars(ap.parse_args())
    return script_arguments


def plot_stats(
    data_map: Dict[bool, List[int]],
    save_path: str,
    x_label: str,
) -> None:
    """
    Plots the histogram for number of turns/number of tokens.
    Plots it separately for trajectories that reached the goal,
    trajectories that did not reach the goal,
    and combined.

    Input:
        data_map (Dict[bool, List[int]]):
            Number of assistant tokens/
            number of turnsrequired to solve the task,
            based on whether the task is solved or not.

        save_path (str):
            Location to save the plot in.

        x_label (str):
            Label for the x axis

    Output:
        None
    """
    if x_label == "Num turns":
        fig = plt.figure(figsize=(15, 3))
        plot_index = 0
        keys = [True, False, "Combined"]

        for key in [True, False, "Combined"]:
            if key == "Combined":
                data = data_map[True] + data_map[False]
            else:
                data = data_map[key]

            plt.subplot(1, len(keys), plot_index + 1)

            plt.hist(data)
            plt.xlabel(x_label)
            plt.ylabel("Frequency")
            plt.title(f"Goal reached: {key}")
            plot_index += 1

        print("Plotting num turns")

        plt.savefig(
            save_path,
            bbox_inches="tight",
        )

        plt.close(fig=fig)

    else:
        fig = plt.figure(figsize=(5, 3))

        print("Plotting num tokens")

        plt.hist(data_map)
        plt.xlabel(x_label)
        plt.ylabel("Frequency")

        plt.savefig(
            save_path,
            bbox_inches="tight",
        )

        plt.close(fig=fig)


def main() -> None:
    """
    Main entry point for this script.
    """
    config = get_arguments()
    model_names = [
        # "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
    ]

    tokenizers = load_tokenizers(model_names=model_names)

    lengths_map, num_turns_map = get_conversation_lengths_for_all_files(
        dir=config["dir"],
        tokenizers=tokenizers,
    )

    for model_name in model_names:
        print("Model name: ", model_name)
        print("Max length: ", np.max(lengths_map[model_name]), "\n")

    plot_stats(
        data_map=num_turns_map,
        save_path=config["num_turns_plot_save_path"],
        x_label="Num turns",
    )

    plot_stats(
        data_map=lengths_map[model_names[0]],
        save_path=config["num_tokens_plot_save_path"],
        x_label="Num tokens",
    )


if __name__ == "__main__":
    main()
