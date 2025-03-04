from typing import (
    List,
    Dict,
    Optional,
)
import time
import re

from llm_exploration.inference.inference_engine import LLMInferenceEngine

MASTERMIND_SECRET_CODE_LENGTH = 4


class MastermindInferenceEngine(LLMInferenceEngine):
    """
    This class contains the logic for generating feedback for the
    mastermind game (https://en.wikipedia.org/wiki/Mastermind_(board_game)),
    given a guess of the secret 4-digit code by the LLM or player.
    """

    def __init__(self, mode: str):
        """
        Initializes an object of this class

        Input:
            mode (str):
                The mode for giving feedback on the Mastermind game.
                Available options: "judge" and "env"
        """
        self.mode = mode
        assert self.mode in ["judge", "env"]

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        time.sleep(0.1)

        assert conv[0]["role"] == "system"
        secret_code: str = conv[0]["content"].strip().lower()
        assert len(secret_code) == MASTERMIND_SECRET_CODE_LENGTH and secret_code.isdigit()

        if self.mode == "env":
            assert conv[-1]["role"] == "user"
            last_code_guess_raw = conv[-1]["content"].strip().lower()
            last_code_guess = "".join(last_code_guess_raw.split())

            feedback = ""
            if len(last_code_guess) != MASTERMIND_SECRET_CODE_LENGTH:
                feedback = (
                    f"The code you guessed does not have 4 digits, "
                    f"but has {len(last_code_guess)}. "
                    f"According to the rule of the game, you must generate a "
                    f"4 digit guess."
                )

            elif not last_code_guess.isdigit():
                feedback = (
                    f"The code you guessed does not consist of 4 digits only, "
                    f"please guess a new code!"
                )

            elif last_code_guess == secret_code:
                feedback = "Goal reached"

            else:
                # count exact matches
                num_exact_matches = 0

                for index in range(len(last_code_guess)):
                    if last_code_guess[index] == secret_code[index]:
                        num_exact_matches += 1

                # Count partial matches
                # Create frequency counts of unmatched digits in secret_code and guess
                secret_unmatched = {}
                guess_unmatched = {}

                for s, g in zip(secret_code, last_code_guess):
                    if s != g:  # Only consider non-matching positions
                        secret_unmatched[s] = secret_unmatched.get(s, 0) + 1
                        guess_unmatched[g] = guess_unmatched.get(g, 0) + 1

                # Calculate partial matches by comparing counts of unmatched digits
                partial_matches = 0
                for digit in guess_unmatched:
                    partial_matches += min(
                        secret_unmatched.get(digit, 0),
                        guess_unmatched[digit],
                    )

                feedback = (
                    f"Your last guess has "
                    f"{num_exact_matches} exact matches with the secret code. "
                    f"In other words, exactly {num_exact_matches} digit(s) in your last guess, "
                    f"{last_code_guess_raw}, "
                    f"are in the correct position in the secret code. "
                    f"(We won't reveal the particular digits within your guess "
                    f"that are exact matches, they can be any digit within your guess) "
                    f"Your last guess also has {partial_matches} partial matches. "
                    f"In other words, {partial_matches} digits in your guess, "
                    f"{last_code_guess_raw}, "
                    f"are in the secret code, but in the wrong position. "
                    f"(We won't reveal which digits within your guess are "
                    f"partial matches, they can be any, you must deduce them with reasoning "
                    f"and further guesses and feedbacks.)"
                )

        else:
            assert conv[-1]["role"] == "user"
            last_code_guess = conv[-1]["content"].strip().lower()

            match = re.search(r"<answer>(.*?)</answer>", last_code_guess, re.DOTALL)
            if match:
                last_code_guess = match.group(1).strip().lower()
            else:
                raise ValueError(f"Given response is invalid for Mastermind.")

            if last_code_guess == secret_code:
                feedback = "<VALID>"
            else:
                feedback = "<NOTVALID>"

        return feedback.strip()

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        feedbacks = []

        for index in range(len(convs)):
            feedback = self.generate(
                conv=convs[index],
                max_n_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
            )
            feedbacks.append(feedback)

        return feedbacks
