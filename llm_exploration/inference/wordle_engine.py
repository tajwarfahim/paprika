from typing import (
    List,
    Dict,
    Optional,
)
import re
import time

from llm_exploration.inference.inference_engine import LLMInferenceEngine


WORDLE_WORD_LENGTH = 5


class WordleInferenceEngine(LLMInferenceEngine):
    letter_positions = {
        0: "First",
        1: "Second",
        2: "Third",
        3: "Fourth",
        4: "Fifth",
    }

    """
    This class contains the logic for generating feedback for the 
    Wordle game (https://en.wikipedia.org/wiki/Wordle),
    given a guess of the secret word by the LLM or player.
    """

    def __init__(
        self,
        mode: str,
    ):
        """
        Initializes the Wordle Inference Engine.
        This can have two modes, 'judge' or 'env',
        and it changes the type of the feedback generated.

        NOTE: We write it this way because we saw gpt-4o and gpt-4o-mini
        is inconsistent in generating accurate feedback for the game
        of Wordle.

        Input:
            mode (str):
                Which mode we are using this inference engine for
                Valid modes are "judge" and "env"
        """
        assert mode in ["judge", "env"]
        self.mode = mode

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
        correct_word = conv[0]["content"].strip().lower()
        assert len(correct_word) == WORDLE_WORD_LENGTH

        if self.mode == "env":
            assert conv[-1]["role"] == "user"
            last_word = conv[-1]["content"].strip().lower()

            feedback = ""

            if len(last_word) != WORDLE_WORD_LENGTH:
                feedback = (
                    f"The word you guessed, {last_word}, does not have five letters, "
                    f"but has {len(last_word)}. "
                    f"It is thus not a correct guess and does not follow the rules of "
                    f"the game wordle. Please make another guess!"
                )

            elif last_word == correct_word:
                feedback = "Goal reached"

            else:
                for index in range(len(last_word)):
                    if last_word[index] == correct_word[index]:
                        feedback += (
                            f"{self.letter_positions[index]} letter, "
                            f"{last_word[index]}, is correct and "
                            f"in the correct position in the target word \n"
                        )

                    elif last_word[index] in correct_word:
                        feedback += (
                            f"{self.letter_positions[index]} letter, "
                            f"{last_word[index]}, exists in the "
                            f"target word but in a different position \n"
                        )

                    else:
                        feedback += (
                            f"{self.letter_positions[index]} letter, "
                            f"{last_word[index]}, is not "
                            f"in the target word \n"
                        )

        elif self.mode == "judge":
            assert conv[-1]["role"] == "user"
            last_word = conv[-1]["content"]
            match = re.search(r"<answer>(.*?)</answer>", last_word, re.DOTALL)
            if match:
                last_word = match.group(1).strip().lower()
            else:
                raise ValueError(f"Given response is invalid for Wordle.")

            if last_word == correct_word:
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
