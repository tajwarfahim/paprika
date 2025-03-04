import torch
import os
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from transformers import (
    AutoTokenizer,
)
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

from llm_exploration.common.tokenizer_separators import TokenizerSeparators
from llm_exploration.llm_finetuning.sft_dataset import (
    MultiturnSFTDatasetWithoutTokenization,
)
from llm_exploration.llm_finetuning.dpo_dataset import (
    MultiturnDPODataset,
    MultiturnDPODatasetFromTensors,
)
from llm_exploration.utils.data_utils import (
    read_json,
    write_json,
)


CONVERSATION = List[Dict[str, str]]


def get_game_dataset(
    trainer_type: str,
    data_dir: str,
    dataset_path: Optional[str],
    tokenizer: AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
    ignore_token_id: int,
    rejected_trajectory_sampling_strategy: str,
    threshold: int,
    judge_label_strategy: str,
    num_samples: Optional[int] = None,
    token_length_threshold: int = 2500,
) -> Dataset:
    """
    Helper function to load the correct dataset,
    used for training LLMs.

    Input:
        trainer_type (str):
            The particular training algorithm to be used.
            For example, SFT or DPO

        data_dir (str):
            The directory where data files are stored.
            Usually the parent directory, contains json files
            each containing separate trajectories.

        dataset_path (str):
            Path to the dataset, usually in ".pt" or pytorch tensor format

            NOTE: This is usually only used for the DPO dataset.
            For DPO, we often precompute input_ids, labels and reference model
            log probabilities, so that we do not need to load and store the
            reference model during training.

            If dataset_path is None, we load the individual json files in
            the data_dir, and use them. Otherwise we use the tensors saved
            in dataset_path.

        tokenizer (AutoTokenizer):
            tokenizer to be used to compute input_ids from text trajectories.

        tokenizer_separator (TokenizerSeparators):
            tokenizer separator, used to handle special tokens and/or assistant
            tokens

        ignore_token_id (int):
            The token id we use to mask non-assistant tokens, since we would want
            to ignore non-assistant tokens from loss calculation.
            Usually -100

        rejected_trajectory_sampling_strategy (str):
            Given K trajectories for the same goal, this string determines how
            we create the (chosen, rejected) pair of trajectories, typically
            needed for DPO and similar preference optimization methods.

        threshold (int):
            Minimum difference in number of turns between chosen and rejected
            trajectories

            NOTE: chosen trajectories are typically always successful,
            but if the difference in number of turns between the best trajectory
            (which is successful and solves the task at minimum number of turns)
            and another successful trajectory is greather than or equal to
            threshold, we consider it a rejected trajectory.

        judge_label_strategy (str):
            How to use the judge label in determining how to use a given trajectory
            Current choices supported:
                "count_invalids_as_successes" --> Invalid i.e., judge_label = False,
                                                  but env_label = True is considered a success

                "count_invalids_as_failures" --> Invalid i.e., judge_label = False,
                                                  but env_label = True is considered a failure

                "disregard_invalids" ---> Invalid i.e., judge_label = False, but env_label = True
                                            Are thrown out, i.e., counted as neither success
                                            nor failure

        num_samples (int):
            In case we want to take a sub-sample of examples from the entire dataset,
            this determines the sample size.

        token_length_threshold (int):
            This argument lets us filter away any trajectory that has more tokens than
            a certain threshold.
    """
    if trainer_type in ["SFT", "GEM"]:
        return GameSFTDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
            threshold=threshold,
            judge_label_strategy=judge_label_strategy,
            num_samples=num_samples,
            token_length_threshold=token_length_threshold,
        )

    elif trainer_type in ["DPO", "SimPO", "RegularizedDPO", "LengthControlledDPO"]:
        if dataset_path is None:
            return GameDPODataset(
                data_dir=data_dir,
                tokenizer=tokenizer,
                tokenizer_separator=tokenizer_separator,
                ignore_token_id=ignore_token_id,
                rejected_trajectory_sampling_strategy=rejected_trajectory_sampling_strategy,
                threshold=threshold,
                judge_label_strategy=judge_label_strategy,
                num_samples=num_samples,
                token_length_threshold=token_length_threshold,
            )
        else:
            return GameDPODatasetFromTensors(
                dataset_path=dataset_path,
                num_samples=num_samples,
            )
    else:
        raise ValueError(f"Given trainer type {trainer_type} not supported.")


class GameSFTDataset(MultiturnSFTDatasetWithoutTokenization):
    """
    Defines a SFT dataset over game trajectories.

    NOTE: For general game playing, we typically have K trajectories per game scenario.
    We only pick the best/preferred trajectory per game scenario in
    order to perform supervised fine-tuning.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: AutoTokenizer,
        tokenizer_separator: TokenizerSeparators,
        ignore_token_id: int,
        threshold: int,
        judge_label_strategy: str,
        token_length_threshold: int,
        num_samples: Optional[int] = None,
    ):
        """
        Input:
            data_dir (str):
                The directory where the data (usually in json files)
                is stored. We read each json file and retrieve the trajectories.

            tokenizer (AutoTokenizer):
                tokenizer to be used to compute input_ids from text trajectories.

            tokenizer_separator (TokenizerSeparators):
                tokenizer separator, used to handle special tokens and/or assistant
                tokens

            ignore_token_id (int):
                The token id we use to mask non-assistant tokens, since we would want
                to ignore non-assistant tokens from loss calculation.
                Usually -100

            threshold (int):
                Minimum difference in number of turns between chosen and rejected
                trajectories

                NOTE: chosen trajectories are typically always successful,
                but if the difference in number of turns between the best trajectory
                (which is successful and solves the task at minimum number of turns)
                and another successful trajectory is greather than or equal to
                threshold, we consider it a rejected trajectory.

                NOTE 2: For SFT, we do not run SFT on rejected trajectories

            judge_label_strategy (str):
                How to use the judge label in determining how to use a given trajectory
                Current choices supported:
                    "count_invalids_as_successes" --> Invalid i.e., judge_label = False,
                                                    but env_label = True is considered a success

                    "count_invalids_as_failures" --> Invalid i.e., judge_label = False,
                                                    but env_label = True is considered a failure

                    "disregard_invalids" ---> Invalid i.e., judge_label = False, but env_label = True
                                                Are thrown out, i.e., counted as neither success
                                                nor failure

            token_length_threshold (int):
                This argument lets us filter away any trajectory that has more tokens than
                a certain threshold.

            num_samples (int):
                In case we want to take a sub-sample of examples from the entire dataset,
                this determines the sample size.
        """
        successful_trajectories = load_game_trajectories(
            data_dir=data_dir,
            rejected_trajectory_sampling_strategy=None,
            threshold=threshold,
            judge_label_strategy=judge_label_strategy,
            load_rejected_trajectories=False,
            tokenizer=tokenizer,
            token_length_threshold=token_length_threshold,
        )

        super().__init__(
            conversations=successful_trajectories,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
            max_token_length=token_length_threshold,
        )

        if num_samples is None:
            self.indices = [i for i in range(super().__len__())]
        else:
            if not isinstance(num_samples, int) or num_samples <= 0:
                raise ValueError(f"Given num samples {num_samples} is not valid.")

            self.indices = np.random.choice(
                a=super().__len__(),
                size=num_samples,
                replace=(num_samples > super().__len__()),
            ).tolist()

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the i-th datapoint in a particular format.

        NOTE: There is an additional remapping function
        because of the support for subsampling the original set
        of trajectories.

        Input:
            idx (int):
                The index of the datapoint that needs to be returned.

        Output:
            datapoint_dict (Dict[str, torch.Tensor]):
                The i-th datapoint in the following dictionary format:
                    {
                        "input_ids": input_id (tensor),
                        "labels": label (tensor),
                        "attention_mask": attention_mask (tensor),
                    }
        """
        remapped_idx = self.indices[idx]
        return super().__getitem__(idx=remapped_idx)


class GameDPODataset(MultiturnDPODataset):
    """
    Defines a DPO dataset over game trajectories.

    NOTE: Unlike GameDPODatasetFromTensors, this class considers
    data to be stored in json files in the given data_dir, and this class
    retrieves the trajectories from each file.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: AutoTokenizer,
        tokenizer_separator: TokenizerSeparators,
        ignore_token_id: int,
        rejected_trajectory_sampling_strategy: str,
        threshold: int,
        judge_label_strategy: str,
        token_length_threshold: int,
        num_samples: Optional[int] = None,
    ):
        """
        Input:
            data_dir (str):
                The directory where the data (usually in json files)
                is stored. We read each json file and retrieve the trajectories.

            tokenizer (AutoTokenizer):
                tokenizer to be used to compute input_ids from text trajectories.

            tokenizer_separator (TokenizerSeparators):
                tokenizer separator, used to handle special tokens and/or assistant
                tokens

            ignore_token_id (int):
                The token id we use to mask non-assistant tokens, since we would want
                to ignore non-assistant tokens from loss calculation.
                Usually -100

            rejected_trajectory_sampling_strategy (str):
                Given K trajectories for the same goal, this string determines how
                we create the (chosen, rejected) pair of trajectories, typically
                needed for DPO and similar preference optimization methods.

            threshold (int):
                Minimum difference in number of turns between chosen and rejected
                trajectories

                NOTE: chosen trajectories are typically always successful,
                but if the difference in number of turns between the best trajectory
                (which is successful and solves the task at minimum number of turns)
                and another successful trajectory is greather than or equal to
                threshold, we consider it a rejected trajectory.

            judge_label_strategy (str):
                How to use the judge label in determining how to use a given trajectory
                Current choices supported:
                    "count_invalids_as_successes" --> Invalid i.e., judge_label = False,
                                                    but env_label = True is considered a success

                    "count_invalids_as_failures" --> Invalid i.e., judge_label = False,
                                                    but env_label = True is considered a failure

                    "disregard_invalids" ---> Invalid i.e., judge_label = False, but env_label = True
                                                Are thrown out, i.e., counted as neither success
                                                nor failure

            token_length_threshold (int):
                This argument lets us filter away any trajectory that has more tokens than
                a certain threshold.

            num_samples (int):
                In case we want to take a sub-sample of examples from the entire dataset,
                this determines the sample size.
        """
        print("Token length threshold: ", token_length_threshold, "\n")

        chosen_trajectories, rejected_trajectories = load_game_trajectories(
            data_dir=data_dir,
            rejected_trajectory_sampling_strategy=rejected_trajectory_sampling_strategy,
            threshold=threshold,
            judge_label_strategy=judge_label_strategy,
            load_rejected_trajectories=True,
            tokenizer=tokenizer,
            token_length_threshold=token_length_threshold,
        )

        assert len(chosen_trajectories) == len(rejected_trajectories)

        super().__init__(
            chosen_conversations=chosen_trajectories,
            rejected_conversations=rejected_trajectories,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
            max_token_length=token_length_threshold,
        )

        if num_samples is None:
            self.indices = [i for i in range(super().__len__())]
        else:
            if not isinstance(num_samples, int) or num_samples <= 0:
                raise ValueError(f"Given num samples {num_samples} is not valid.")

            self.indices = np.random.choice(
                a=super().__len__(),
                size=num_samples,
                replace=(num_samples > super().__len__()),
            ).tolist()

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the i-th datapoint in a particular format.

        NOTE: There is an additional remapping function
        because of the support for subsampling the original set
        of trajectories.

        Input:
            idx (int):
                The index of the datapoint that needs to be returned.

        Output:
            datapoint_dict (Dict[str, torch.Tensor]):
                The i-th datapoint in the following dictionary format:
                    {
                        "input_ids": input_id (tensor),
                        "labels": label (tensor),
                        "attention_mask": attention_mask (tensor),
                    }
        """
        remapped_idx = self.indices[idx]
        return super().__getitem__(idx=remapped_idx)

    def save_dataset_trajectories_to_json(self, save_path: str) -> None:
        """
        Saves a dataset to the given json file path.
        It saves it in the following format:
        [
            {
                "chosen_trajectory": chosen_trajectory,
                "rejected_trajectory": rejected_trajectory,
            },
            ...
        ]

        where each trajectory looks like the following:
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            ...
        ]

        Input:
            save_path (str):
                The path to save the data

        Output:
            None
        """
        all_data = []
        for idx in range(len(self)):
            remapped_idx = self.indices[idx]
            chosen_trajectory = self.chosen_conversations[remapped_idx]
            rejected_trajectory = self.rejected_conversations[remapped_idx]
            data = {
                "chosen_trajectory": chosen_trajectory,
                "rejected_trajectory": rejected_trajectory,
            }
            all_data.append(data)

        write_json(data=all_data, fname=save_path)


class GameDPODatasetFromTensors(MultiturnDPODatasetFromTensors):
    """
    Specific implementation of the DPO dataset over game trajectories,
    to be able to leverage pretrained log probabilities from the reference
    model.

    Specifically, a separate script calculates the log probabilities from
    the reference model and stores it in dataset_path (in ".pt" format)

    This class uses the pre-computed tensors.
    """

    def __init__(
        self,
        dataset_path: str,
        num_samples: Optional[int] = None,
    ):
        """
        Input:
            dataset_path (str):
                Path to the dataset, usually in ".pt" or pytorch tensor format

                NOTE: This is usually only used for the DPO dataset.
                For DPO, we often precompute input_ids, labels and reference model
                log probabilities, so that we do not need to load and store the
                reference model during training.

                If dataset_path is None, we load the individual json files in
                the data_dir, and use them. Otherwise we use the tensors saved
                in dataset_path.

            num_samples (int):
                In case we want to take a sub-sample of examples from the entire dataset,
                this determines the sample size.
        """
        super().__init__(dataset_path=dataset_path)

        if num_samples is None:
            self.indices = [i for i in range(super().__len__())]
        else:
            if not isinstance(num_samples, int) or num_samples <= 0:
                raise ValueError(f"Given num samples {num_samples} is not valid.")

            self.indices = np.random.choice(
                a=super().__len__(),
                size=num_samples,
                replace=(num_samples > super().__len__()),
            ).tolist()

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the i-th datapoint in a particular format.

        NOTE: There is an additional remapping function
        because of the support for subsampling the original set
        of trajectories.

        Input:
            idx (int):
                The index of the datapoint that needs to be returned.

        Output:
            datapoint_dict (Dict[str, torch.Tensor]):
                The i-th datapoint in the following dictionary format:
                    {
                        "input_ids": input_id (tensor),
                        "labels": label (tensor),
                        "attention_mask": attention_mask (tensor),
                    }
        """
        remapped_idx = self.indices[idx]
        return super().__getitem__(idx=remapped_idx)


def load_game_trajectories(
    data_dir: str,
    rejected_trajectory_sampling_strategy: str,
    threshold: int,
    judge_label_strategy: str,
    load_rejected_trajectories: bool,
    tokenizer: AutoTokenizer,
    token_length_threshold: int,
) -> Union[Tuple[List[CONVERSATION], List[CONVERSATION]], List[CONVERSATION]]:
    """
    Input:
        data_dir (str):
            The directory where the data (usually in json files)
            is stored. We read each json file and retrieve the trajectories.

        rejected_trajectory_sampling_strategy (str):
            Given K trajectories for the same goal, this string determines how
            we create the (chosen, rejected) pair of trajectories, typically
            needed for DPO and similar preference optimization methods.

        threshold (int):
            Minimum difference in number of turns between chosen and rejected
            trajectories

        judge_label_strategy (str):
            How to use the judge label in determining how to use a given trajectory
            Current choices supported:
                "count_invalids_as_successes" --> Invalid i.e., judge_label = False,
                                                  but env_label = True is considered a success

                "count_invalids_as_failures" --> Invalid i.e., judge_label = False,
                                                  but env_label = True is considered a failure

                "disregard_invalids" ---> Invalid i.e., judge_label = False, but env_label = True
                                            Are thrown out, i.e., counted as neither success
                                            nor failure

        load_rejected_trajectories (bool):
            Whether to load rejected (sub-optimal) trajectories or not.
            If True, we return (chosen_trajectories, rejected_trajectories)
            If False, we return chosen_trajectories only

        tokenizer (AutoTokenizer):
            tokenizer to be used to compute input_ids from text trajectories.

        token_length_threshold (int):
            This argument lets us filter away any trajectory that has more tokens than
            a certain threshold.

    Output:
        Either
            1. (chosen_trajectories, rejected_trajectories) tuple, OR
            2. chosen_trajectories

        depending on value of load_rejected_trajectories that is passed as parameter

        chosen_trajectories (List[List[Dict[str, str]]]):
            chosen_trajectories[i] is the preferred trajectory for the i-th game scenario

        rejected_trajectories (List[List[Dict[str, str]]]):
            rejected_trajectories[i] is the dispreferred/rejected trajectory
            for the i-th game scenario

        NOTE:
            1. Each trajectory should look like the following:
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ...
            ]

            2. If for a given game scenario, none of the K trajectories are successful,
            then we do not include it in the dataset.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Give data directory {data_dir} is invalid.")

    files = [
        os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".json")
    ]
    files = sorted(files)

    # ensures we group by game scenario,
    # since same game scenario can appear in different files,
    # so we do the grouping first
    game_scenario_to_trial_map = defaultdict(list)
    for file_index in tqdm(range(len(files))):
        file = files[file_index]
        data = read_json(fname=file)
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            raise ValueError(f"Data file is corrupted.")

        for record in data["records"]:
            if not isinstance(record, list):
                raise ValueError(f"Data file is corrupted")

            for i in range(len(record)):
                trial = record[i]
                if (
                    not isinstance(trial, dict)
                    or not isinstance(trial.get("conversation"), list)
                    or not isinstance(trial.get("env_game_scenario"), str)
                ):
                    raise ValueError(f"Data file is corrupted")

                env_game_scenario = trial["env_game_scenario"]
                conversation = trial["conversation"]

                tokenized_conversation = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                )

                if len(tokenized_conversation) <= token_length_threshold:
                    game_scenario_to_trial_map[env_game_scenario].append(trial)

    # make the list of chosen and (optionally) rejected trajectories
    chosen_trajectories = []

    if load_rejected_trajectories:
        rejected_trajectories = []

    for game_scenario in game_scenario_to_trial_map:
        record = game_scenario_to_trial_map[game_scenario]

        # get all the data for a single game scenario
        all_data_per_goal = []
        for i in range(len(record)):
            trial = record[i]

            conversation = trial["conversation"]
            for index in range(len(conversation)):
                conversation[index]["content"] = conversation[index]["content"].strip()

            if judge_label_strategy == "count_invalids_as_successes":
                judge_label = True
                env_label = trial["goal_reached"]

            elif judge_label_strategy == "count_invalids_as_failures":
                judge_label = True
                env_label = trial["goal_reached"] and trial["judge_label"]

            elif judge_label_strategy == "disregard_invalids":
                judge_label = trial["judge_label"]
                env_label = trial["goal_reached"]

            else:
                raise ValueError(f"Given strategy {judge_label_strategy} not supported.")

            all_data_per_goal.append(
                [trial["num_turns"], env_label, judge_label, conversation]
            )

        # choose the chosen and rejected trajectory for the current game scenario
        if load_rejected_trajectories:
            (
                chosen_trajectories_per_goal,
                rejected_trajectories_per_goal,
            ) = get_chosen_and_rejected_trajectories(
                all_data_per_goal=all_data_per_goal,
                rejected_trajectory_sampling_strategy=rejected_trajectory_sampling_strategy,
                threshold=threshold,
            )

            # in case we did not have a chosen and rejected trajectory,
            # we do not add it to the dataset
            if (
                chosen_trajectories_per_goal is not None
                and rejected_trajectories_per_goal is not None
            ):
                chosen_trajectories.extend(chosen_trajectories_per_goal)
                rejected_trajectories.extend(rejected_trajectories_per_goal)

        else:
            chosen_trajectories_per_goal = get_only_successful_trajectories(
                all_data_per_goal=all_data_per_goal,
                threshold=threshold,
            )

            if chosen_trajectories_per_goal is not None:
                chosen_trajectories.extend(chosen_trajectories_per_goal)

    if load_rejected_trajectories:
        return chosen_trajectories, rejected_trajectories
    else:
        return chosen_trajectories


def validate_trajectory(
    trajectory: List[Dict[str, str]],
) -> None:
    """
    Validate whether a given trajectory has the correct form,
    which looks like below:
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
            ...
        ]

    Input:
        trajectory (List[Dict[str, str]]):
            A game trajectory

    Output:
        None
    """
    assert isinstance(trajectory, list)

    for item in trajectory:
        assert isinstance(item, dict)

        for key in item:
            assert key in ["role", "content"]
            assert isinstance(item[key], str)

        assert item.get("role") in ["system", "user", "assistant"]


def prepare_data_per_goal(all_data_per_goal: List) -> List:
    """
    Sorts all the trajectories per goal/game scenario.
    Sorting criteria:
        1. First sort by game success: successful is preferred
        2. Next, among successful trajectories, fewer number of turns
           is better

    We also filter by judge label, throwing out (not even keeping
    in the list) trajectories if the judge considers them "invalid",
    despite the env label on success/failure.

    In the final output, judge label is no longer included

    Input:
        all_data_per_goal (List):
            All the data for a given goal/game scenario.
            all_data_per_goal[i] = the i-th datapoint, which is a List of
                                    [
                                        num turns required,
                                        whether env things game is solved,
                                        whether judge things game is solved,
                                        trajectory,
                                    ]

    Output:
        modified_all_data_per_goal (List):
            All the data for a given goal/game scenario.
            all_data_per_goal[i] = the i-th datapoint, which is a List of
                                    [
                                        num turns required,
                                        whether env things game is solved,
                                        trajectory,
                                    ]
    """
    # we remove all invalid trajectories according to the judge
    all_data_per_goal_filtered_by_judge = []
    for i in range(len(all_data_per_goal)):
        if all_data_per_goal[i][2]:
            new_data = [
                all_data_per_goal[i][0],
                all_data_per_goal[i][1],
                all_data_per_goal[i][3],
            ]
            all_data_per_goal_filtered_by_judge.append(new_data)

    all_data_per_goal = deepcopy(all_data_per_goal_filtered_by_judge)

    # data = [num turns required, env label, trajectory] Lists
    # we want to sort first by whether the game is solved, and then by number of turns
    for i in range(len(all_data_per_goal)):
        if len(all_data_per_goal[i]) != 3:
            print(all_data_per_goal[i])
            raise ValueError(f"Data not in correct format.")

        all_data_per_goal[i][1] = 0 if all_data_per_goal[i][1] else 1

    return sorted(all_data_per_goal, key=lambda x: (x[1], x[0]))


def get_only_successful_trajectories(
    all_data_per_goal: List,
    threshold: int,
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """
    Loads only successful trajectories from a list of trajectories,
    all sampled for the same goal or game scenario.

    We consider a trajectory successful if:
        1. The trajectory has env_label = True (after counting for judge label)
        2. The trajectory has num turns < best num turns + threshold

    Input:
        all_data_per_goal (List):
            All the data for a given goal/game scenario.
            all_data_per_goal[i] = the i-th datapoint, which is a List of
                                    [
                                        num turns required,
                                        whether env things game is solved,
                                        whether judge things game is solved,
                                        trajectory,
                                    ]

        threshold (int):
            The maximum difference between best num turns and trajectory num turns
            such that a trajectory may still be considered successful.

            NOTE: the bound is non-inclusive
    """
    all_data_per_goal = prepare_data_per_goal(all_data_per_goal=all_data_per_goal)
    # If all trajectories are rejected/failed to solve the game, we skip this game scenario
    if (
        len(all_data_per_goal) == 0
        or len(all_data_per_goal[0]) == 0
        or all_data_per_goal[0][1] == 1
    ):
        return None

    chosen_trajectories = []

    num_turns_previous_trajectory = None
    best_num_turns = all_data_per_goal[0][0]

    for index in range(len(all_data_per_goal)):
        is_trajectory_successful = all_data_per_goal[index][1] == 0
        if not is_trajectory_successful:
            break

        if (
            num_turns_previous_trajectory is not None
            and all_data_per_goal[index][0] == num_turns_previous_trajectory
        ):
            continue

        if all_data_per_goal[index][0] - best_num_turns >= threshold:
            break

        else:
            chosen_trajectories.append(all_data_per_goal[index][-1])
            num_turns_previous_trajectory = all_data_per_goal[index][0]

    if len(chosen_trajectories) == 0:
        return None

    for trajectory in chosen_trajectories:
        validate_trajectory(trajectory=trajectory)

    return chosen_trajectories


def get_chosen_and_rejected_trajectories(
    all_data_per_goal: List,
    rejected_trajectory_sampling_strategy: str,
    threshold: int,
) -> Tuple[List[CONVERSATION], List[CONVERSATION]]:
    """
    For a single goal/game scenario, we have K trajectories
    This function returns the chosen and rejected trajectories
    for this game scenario.

    chosen_trajectory   --> The trajectory that has solved the task
                            AND takes lower number of turns
    rejected_trajectory --> Another trajectory, chosen according to
                            given sampling strategy

    Input:
        all_data_per_goal (List):
            All the data for a given goal/game scenario.
            all_data_per_goal[i] = the i-th datapoint, which is a List of
                                    [
                                        num turns required,
                                        whether env things game is solved,
                                        whether judge things game is solved,
                                        trajectory,
                                    ]

        rejected_trajectory_sampling_strategy (str):
            Given K trajectories for the same goal, we typically choose the best
            one as the preferred trajectory, for running DPO.
            But the rejected trajectory can be chosen to be random/worst/second best
            among the remaining trajectories.

            There are 4 possible choices currently supported.
            (NOTE: All data points are sorted according to num turns required to solve)

            random --> We randomly choose one of the K - 1 other trajectories as rejected
            worst  --> The worst data point among the K data points
            second_best --> Second best trajectory in the list of K trajectories
            all ->
                   1. We collect the best trajectory, and any other successful trajectory
                   with num turns less than (best num turn + threshold) in Good set G

                   2. We collect the bad trajectories, i.e., unsuccessful trajectories
                   and any successful trajectory taking greater than or equal
                   (best num turn + threshold) in bad set B

                   3. For every t in G:
                        Randomly sample t_b in B
                        Pair (t, t_b) and add to dataset

        threshold (int):
            The difference between chosen and rejected trajectory's number of turns

    Output:
        (chosen_trajectories, rejected_trajectories) tuple.

        chosen_trajectories (List[List[Dict[str, str]]]):
            the preferred trajectories

        rejected_trajectories (List[List[Dict[str, str]]]):
            the rejected trajectories, sampled according to the above discussed rule
    """
    all_data_per_goal = prepare_data_per_goal(all_data_per_goal=all_data_per_goal)

    # If all trajectories are rejected/failed to solve the game, we skip this game scenario
    if (
        len(all_data_per_goal) == 0
        or len(all_data_per_goal[0]) == 0
        or all_data_per_goal[0][1] == 1
    ):
        return None, None

    # The chosen trajectory is the one that both
    # 1. solves the game
    # 2. does it in the fewest number of turns
    # Randomly sample one of the other K - 1 trajectories as the rejected one
    if rejected_trajectory_sampling_strategy == "random":
        if len(all_data_per_goal) <= 1:
            raise ValueError(
                f"Only one data point per goal is available, not compatible with "
                f"the rejected sampling strategy {rejected_trajectory_sampling_strategy}"
            )

        chosen_trajectories = [all_data_per_goal[0][-1]]
        random_index = np.random.choice([i for i in range(1, len(all_data_per_goal))])
        rejected_trajectories = [all_data_per_goal[random_index]]

        is_rejected_trajectory_successful = all_data_per_goal[random_index][1] == 0
        chosen_num_turns = all_data_per_goal[0][0]
        rejected_num_turns = all_data_per_goal[random_index][0]

        # If both chosen and rejected trajectory are both successful and have the same length
        # then they are technically tied, and we don't include this game scenario in the dataset
        if is_rejected_trajectory_successful and chosen_num_turns == rejected_num_turns:
            return None, None

    # The chosen trajectory is the one that both
    # 1. solves the game
    # 2. does it in the fewest number of turns
    # Pick the worst trajectory as the rejected one
    elif rejected_trajectory_sampling_strategy == "worst":
        if len(all_data_per_goal) <= 1:
            raise ValueError(
                f"Only one data point per goal is available, not compatible with "
                f"the rejected sampling strategy {rejected_trajectory_sampling_strategy}"
            )

        chosen_trajectories = [all_data_per_goal[0][-1]]
        rejected_trajectories = [all_data_per_goal[-1][-1]]

        is_rejected_trajectory_successful = all_data_per_goal[-1][1] == 0
        chosen_num_turns = all_data_per_goal[0][0]
        rejected_num_turns = all_data_per_goal[-1][0]

        if is_rejected_trajectory_successful and chosen_num_turns == rejected_num_turns:
            return None, None

    # The chosen trajectory is the one that both
    # 1. solves the game
    # 2. does it in the fewest number of turns
    # Pick the second best trajectory as the rejected one
    elif rejected_trajectory_sampling_strategy == "second_best":
        if len(all_data_per_goal) <= 1:
            raise ValueError(
                f"Only one data point per goal is available, not compatible with "
                f"the rejected sampling strategy {rejected_trajectory_sampling_strategy}"
            )

        chosen_trajectories = [all_data_per_goal[0][-1]]
        rejected_trajectories = [all_data_per_goal[1][-1]]

        is_rejected_trajectory_successful = all_data_per_goal[1][1] == 0
        chosen_num_turns = all_data_per_goal[0][0]
        rejected_num_turns = all_data_per_goal[1][0]

        if is_rejected_trajectory_successful and chosen_num_turns == rejected_num_turns:
            return None, None

    # Sample all possible (chosen, rejected) pairs with some constraints
    elif rejected_trajectory_sampling_strategy == "all":
        chosen_trajectories = []
        rejected_trajectories = []

        chosen_indices = []
        num_turns_previous_trajectory = None
        best_num_turns = all_data_per_goal[0][0]

        for index in range(len(all_data_per_goal)):
            is_trajectory_successful = all_data_per_goal[index][1] == 0
            if not is_trajectory_successful:
                break

            if (
                num_turns_previous_trajectory is not None
                and all_data_per_goal[index][0] == num_turns_previous_trajectory
            ):
                continue

            if all_data_per_goal[index][0] - best_num_turns >= threshold:
                break

            else:
                chosen_indices.append(index)
                num_turns_previous_trajectory = all_data_per_goal[index][0]

        if len(chosen_indices) == 0:
            return None, None

        worst_num_turns_among_chosen = all_data_per_goal[chosen_indices[-1]][0]

        # By the way we have sorted the data,
        # we don't need the next code block --- just here as an extra check!
        rejected_indices = []
        for index in range(max(chosen_indices) + 1, len(all_data_per_goal)):
            is_trajectory_successful = all_data_per_goal[index][1] == 0
            num_turns = all_data_per_goal[index][0]

            if not is_trajectory_successful:
                rejected_indices.append(index)
            elif num_turns - worst_num_turns_among_chosen >= threshold:
                rejected_indices.append(index)

        has_rejected_trajectories = len(rejected_indices) > 0

        for chosen_index in chosen_indices:
            # If there are no rejected indices, we consider the chosen trajectory
            # as the rejected trajectory. The trainer code is rewritten to
            # ignore any DPO-like contrastive loss on these trajectories
            if has_rejected_trajectories:
                rejected_index = np.random.choice(a=rejected_indices)
            else:
                rejected_index = chosen_index

            chosen_trajectory_candidate = all_data_per_goal[chosen_index][-1]
            rejected_trajectory_candidate = all_data_per_goal[rejected_index][-1]

            chosen_trajectories.append(chosen_trajectory_candidate)
            rejected_trajectories.append(rejected_trajectory_candidate)

            break

    else:
        raise ValueError(
            f"Given rejected trajectory sampling strategy"
            f"{rejected_trajectory_sampling_strategy} is not supported."
        )

    for trajectory in chosen_trajectories + rejected_trajectories:
        validate_trajectory(trajectory=trajectory)

    return chosen_trajectories, rejected_trajectories
