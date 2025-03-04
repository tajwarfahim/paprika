"""
This file implements elementary (1D) cellular automations,
as described here: 
https://en.wikipedia.org/wiki/Elementary_cellular_automaton
"""

from typing import (
    List,
    Dict,
    Optional,
)
import time

from llm_exploration.inference.inference_engine import LLMInferenceEngine
from llm_exploration.game.cellular_automation import (
    generate_next_state_1D_cellular_automatation,
)


def extract_inputs(text: str) -> List[str]:
    """
    Given a string of a particular format, extracts
    all the inputs, and returns them.

    Input:
        text (str):
            Usually has the following format:
            "
            Input 1: ....
            Output 1: ....

            Input 2: ....
            Output 2: ....
            "

    Output:
        all_input_list (List[str]):
            List of all the inputs in the text string
    """
    input_num = 1
    all_input_list = []

    while True:
        if f"Input {input_num}:" not in text or f"Output {input_num}:" not in text:
            break

        input_i = text.split(f"Input {input_num}:")[-1]
        input_i = input_i.split(f"Output {input_num}:")[0].strip()

        all_input_list.append(input_i)
        input_num += 1

    return all_input_list


def extract_outputs(text: str) -> List[str]:
    """
    Given a string of a particular format, extracts
    all the outputs, and returns them.

    Input:
        text (str):
            Usually has the following format:
            "
            Input 1: ....
            Output 1: ....

            Input 2: ....
            Output 2: ....
            "

    Output:
        all_output_list (List[str]):
            List of all the outputs in the text string
    """
    output_num = 1
    all_output_list = []

    while True:
        if f"Output {output_num}:" not in text:
            break

        output_i = text.split(f"Output {output_num}:")[-1]
        if f"Input {output_num + 1}:" in output_i:
            output_i = output_i.split(f"Input {output_num + 1}:")[0].strip()

        all_output_list.append(output_i.strip())
        output_num += 1

    return all_output_list


def extract_automation_rule(text: str) -> Dict[str, str]:
    """
    Given a text of the following format:
        "<rule> 111: 1 </rule>\n<rule> 110: 2 </rule> ..."
    This function extracts the automation ruleset from
    this string.

    Input:
        text (str):
            Text that contains (most likely LLM generated)
            automation rule

    Output:
        automation_rule (Dict[str, str]):
            Extracted automation rule
    """
    neighborhoods = [
        "111",
        "110",
        "101",
        "100",
        "011",
        "010",
        "001",
        "000",
    ]

    automation_rule = {}
    for neighborhood in neighborhoods:
        assert neighborhood in text, print(text)
        text_part = text.split(f"{neighborhood}:")[-1]

        assert "</rule>" in text_part, print(text)
        next_cell_state = text_part.split("</rule>")[0].strip()

        assert len(next_cell_state) == 1, print(text)
        automation_rule[neighborhood] = next_cell_state

    return automation_rule


class CellularAutomationInferenceEngine(LLMInferenceEngine):
    def __init__(
        self,
        mode: str,
    ):
        """
        Initializes the Cellular automation Inference Engine.
        This can have two modes, 'judge' or 'env',
        and it changes the type of the feedback generated.

        NOTE: We write it this way because we saw gpt-4o and gpt-4o-mini
        is inconsistent in generating accurate feedback for this game.

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
        all_input_list = extract_inputs(text=conv[0]["content"])
        all_true_output_list = extract_outputs(text=conv[0]["content"])
        assert len(all_input_list) == len(all_true_output_list)

        assert conv[-1]["role"] == "user"
        automation_rule = extract_automation_rule(
            text=conv[-1]["content"],
        )

        outputs_generated_by_given_automation_rule = []
        for input_cell_state in all_input_list:
            output_cell_state = generate_next_state_1D_cellular_automatation(
                current_state=input_cell_state.split(), automation_rule=automation_rule
            )
            output_cell_state = " ".join(output_cell_state)
            outputs_generated_by_given_automation_rule.append(output_cell_state)

        goal_reached = True
        for index in range(len(all_true_output_list)):
            true_output = all_true_output_list[index]
            generated_output = outputs_generated_by_given_automation_rule[index]

            if true_output != generated_output:
                goal_reached = False
                break

        if self.mode == "env":
            if goal_reached:
                feedback = "Goal reached"

            else:
                feedback = (
                    "Sorry, the automation rule you guessed does not "
                    "generate the correct outputs for all the given inputs. "
                    "I will give you the outputs from the rules that you gave last time. "
                    "Please use them to refine your guess about the automation rule."
                )

                for index in range(len(all_input_list)):
                    feedback += f"\n\nInput {index + 1}: " + all_input_list[index]
                    feedback += f"\nTrue Output {index + 1}: " + all_true_output_list[index]
                    feedback += f"\nOutput generated by the last rule you gave: "
                    feedback += outputs_generated_by_given_automation_rule[index]

        else:
            feedback = "<VALID>" if goal_reached else "<NOTVALID>"

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
