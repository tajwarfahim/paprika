import torch
from tqdm import tqdm
from typing import Dict, Any, List
from transformers import AutoTokenizer

from llm_exploration.llm_finetuning.sft_dataset import MultiturnSFTDataset
from llm_exploration.common.tokenizer_separators import TokenizerSeparators


class MultiturnRWRDataset(MultiturnSFTDataset):
    """Reward Weighted Returns (RWR),  an extension of SFT.
    Assumptions:
    1. Each turn in the conversation is assigned a reward.
    2. For conversation-level rewards, only the final turn receives a raw reward. The other turns should have r_min.
    3. For any token in a specific turn, its reward is the "reward-to-go" calculated for that turn.
       The reward-to-go represents the cumulative future rewards from that point onward.

    Note: This implementation supports both conversation-level and potential future
    turn-level reward labeling without requiring changes to this class.
    """

    def __init__(
        self,
        conversations: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        tokenizer_separator: TokenizerSeparators,
        ignore_token_id: int,
        reward_per_turns: List[List[float]],
        gamma: float = 0.9,
        min_reward: float = 0.0,
    ):
        """
        Initializes a dataset of this class.

        Input:
            conversations (List[List[Dict[str, Any]]]):
                List of conversations that will be in this dataset.
                [
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        } ....
                    ], # 1st conversation
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        } ....
                    ], # 2nd conversation
                    .... (More conversations like this)
                ]

            tokenizer (Tokenizer):
                tokenizer for the model to be trained.

            tokenizer_separator (TokenizerSeparators):
                The tokenizer separator for assistant/user special tokens.

            ignore_token_id (int):
                The token that should be ignored from loss calculations.

            reward_per_turns (List[List[float]]):
                reward_per_turns[i][j] = the reward given for the j-th turn,
                                         in the i-th conversation

            gamma (float):
                Discount factor for the typical RL problem.
                Default: 0.9

            min_reward (float):
                The minimum reward to give for every turn,
                if reward is below this, we clip it.
                Default: 0.0
        """
        super().__init__(
            conversations=conversations,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
        )

        self.rewards = []

        for i, reward_per_turn in tqdm(enumerate(reward_per_turns), desc="Rewards - RWR"):
            self.rewards.append(
                get_token_level_reward_to_gos(
                    rewards_per_turn=reward_per_turn,
                    masked_tokens=self.labels[i],
                    ignore_token_id=ignore_token_id,
                    gamma=gamma,
                    min_reward=min_reward,
                )
            )
        self.rewards = torch.stack(self.rewards, dim=0)

    def __len__(self):
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        """
        Returns the i-th datapoint in a particular format.

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
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
            "rewards": self.rewards[idx],
        }


def get_reward_to_gos(rewards: List[float], gamma: float) -> List[float]:
    """Calculate reward to go for each turn.
    Args:
        rewards: List of raw rewards per turn
        gamma: Discount factor to calculate reward to go
    reward_to_go_{i} = \Sum_{j=i}^{n} \gamma^{j-i} * rewards[j]
    """
    reward_to_gos = [0] * len(rewards)
    reward_to_go = 0
    for i in range(len(rewards) - 1, -1, -1):
        reward_to_go = rewards[i] + gamma * reward_to_go
        reward_to_gos[i] = reward_to_go
    return reward_to_gos


def get_token_level_reward_to_gos(
    rewards_per_turn: List[float],
    masked_tokens: torch.Tensor,
    ignore_token_id: Any,
    gamma: float,
    min_reward: float,
) -> torch.Tensor:
    """Calculate reward to go for each token in the conversation.
    Args:
        rewards_per_turn: List of raw rewards per turn
        masked_tokens: Masked tokens in the conversation; All tokens except assistant responses should be masked.
        ignore_token_id: Token id to ignore; depends on label_smoother used; default : -100
        gamma: Discount factor to calculate reward to go
        min_reward: Minimum reward to assign to tokens; All masked tokens should be assigned this reward by default.
    """
    # All assistant responses are unmasked.
    unmasked_turn_indices = torch.where(masked_tokens != ignore_token_id)[0]
    # Calculate differences between consecutive indices
    index_diffs = torch.diff(unmasked_turn_indices)

    # Select turn boundary indices from unmasked_turn_indices [s_0, s_1, .... s_t, e_t]
    turn_boundaries = torch.cat(
        [
            torch.tensor([0]),  # s_0
            torch.where(index_diffs != 1)[0] + 1,  # s_1, s_2, .... s_t
            torch.tensor([len(unmasked_turn_indices)]),  # e_t
        ]
    )
    # Turn masks for each token. -1 for masked tokens.
    # Assistant responses are unmasked -> Assigned turn number.
    turn_masks = torch.full_like(masked_tokens, -1.0)
    for i in range(len(turn_boundaries) - 1):
        start = unmasked_turn_indices[turn_boundaries[i]]
        end = unmasked_turn_indices[turn_boundaries[i + 1] - 1] + 1
        turn_masks[start:end] = i

    assert len(turn_boundaries) - 1 == len(
        rewards_per_turn
    ), "Number of turns should match number of rewards"
    reward_to_gos = get_reward_to_gos(rewards_per_turn, gamma)
    token_rewards = torch.full_like(masked_tokens, min_reward, dtype=torch.float)

    for i in range(len(reward_to_gos)):
        # replace all token_rewards in turn i with reward_to_gos[i]
        token_rewards.masked_fill_(turn_masks.eq(i), reward_to_gos[i])
    return token_rewards
