from datasets import load_dataset
from typing import Dict, Any
from transformers import AutoTokenizer
from tqdm import tqdm

from llm_exploration.utils.data_utils import write_json


def clean_wildchat_dataset() -> Dict[str, Any]:
    dataset = load_dataset(
        "allenai/WildChat-1M",
        split="train",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        use_fast=False,
        trust_remote_code=True,
    )

    records = []
    num_max_turns = 0
    max_sequence_length = 0

    for index in tqdm(range(len(dataset))):
        conversation = dataset[index]["conversation"]
        filtered_conversation = []
        env_game_scenario = dataset[index]["conversation_hash"]
        conversation_is_English = True

        for j in range(len(conversation)):
            turn_data = {
                "role": conversation[j]["role"],
                "content": conversation[j]["content"],
            }
            filtered_conversation.append(turn_data)

            conversation_is_English = (
                conversation_is_English
                and conversation[j].get("language", "English") == "English"
            )

        if conversation_is_English:
            record = {
                "conversation": filtered_conversation,
                "env_game_scenario": env_game_scenario,
                "env_first_message": env_game_scenario,
                "goal_reached": True,
                "judge_label": True,
                "max_turns": dataset[index]["turn"],
                "num_turns": dataset[index]["turn"],
                "agent_game_scenario": env_game_scenario,
            }

            tokenized_seq = tokenizer.apply_chat_template(
                filtered_conversation,
                tokenize=True,
            )

            max_sequence_length = max(
                max_sequence_length,
                len(tokenized_seq),
            )

            num_max_turns = max(num_max_turns, dataset[index]["turn"])

            records.append([record])

    print("\nNumber of maximum turns in dataset:", num_max_turns)
    print("Highest sequence length: ", max_sequence_length)
    print("Number of filtered datapoints: ", len(records), "\n")

    return records


def clean_and_save_wildchat_dataset() -> None:
    records = clean_wildchat_dataset()
    save_path = "/home/ftajwar/training_datasets/wildchat_cleaned_dataset/wildchat_cleaned_dataset.json"

    write_json(
        data={"records": records},
        fname=save_path,
    )


if __name__ == "__main__":
    clean_and_save_wildchat_dataset()
