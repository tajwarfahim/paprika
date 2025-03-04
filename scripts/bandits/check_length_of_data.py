from typing import List, Dict, Any
import os
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
import argparse

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
) -> Dict[str, List[int]]:
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
    """
    assert os.path.isfile(json_file_path) and json_file_path.endswith(".json")
    data = read_json(fname=json_file_path)

    lengths_map = defaultdict(list)
    for record in data["records"]:
        conversation = record["conversation"]

        for model_name in tokenizers:
            tokenizer = tokenizers[model_name]
            tokenized_seq = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
            )
            lengths_map[model_name].append(len(tokenized_seq))

    return lengths_map


def get_conversation_lengths_for_all_files(
    dir: str,
    tokenizers: Dict[str, AutoTokenizer],
) -> Dict[str, List[int]]:
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
    """
    assert os.path.isdir(dir)

    lengths_map = defaultdict(list)

    all_files = os.listdir(dir)
    for file in all_files:
        json_file_path = os.path.join(dir, file)
        if os.path.isfile(json_file_path) and json_file_path.endswith(".json"):
            file_length_map = get_conversation_lengths(
                json_file_path=json_file_path,
                tokenizers=tokenizers,
            )

            for key in file_length_map:
                lengths_map[key].extend(file_length_map[key])

    return lengths_map


def get_arguments() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-dir",
        "--dir",
        type=str,
        default=DATA_DIR,
    )

    script_arguments = vars(ap.parse_args())
    return script_arguments


def main() -> None:
    config = get_arguments()
    model_names = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]

    tokenizers = load_tokenizers(model_names=model_names)

    lengths_map = get_conversation_lengths_for_all_files(
        dir=config["dir"],
        tokenizers=tokenizers,
    )

    for model_name in model_names:
        print("Model name: ", model_name)
        print("Max length: ", np.max(lengths_map[model_name]), "\n")


if __name__ == "__main__":
    main()
