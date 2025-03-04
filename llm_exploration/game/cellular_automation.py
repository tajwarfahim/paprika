from typing import List, Dict


def generate_next_state_1D_cellular_automatation(
    current_state: List[str],
    automation_rule: Dict[str, str],
) -> List[int]:
    """
    Generates the next state from the current state
    in a 1D elementary cellular automation,
    as described here: https://en.wikipedia.org/wiki/Elementary_cellular_automaton

    Input:
        current_state (List[str]):
            The current state of the automation
            Typically would be a list, with each element being '0' or '1'
            (though we don't enforce it, as long as the ruleset supports it)

            Example:
                ['0', '1', '1', '1', '1', '0']

        automation_rule (Dict[str, str]):
            The rules that define the transition
            We consider only a particular cell, its left and right neighbors
            (possibly in a warped style), and use the rule given by automation_rule
            to update it.

            Example:
                {
                    '111': '0',
                    '110': '1',
                    '101': '1',
                    '100': '0',
                    '011': '1',
                    '010': '1',
                    '001': '1',
                    '000': '0',
                }

    Output:
        next_state (List[str]):
            The next state of the automation,
            obtained from the inputs

            We consider a particular cell and its immediate neighbors (potentially
            in a warp-around edge way), to calculate the next state.

            Example:
                Assume the example inputs above. Using the automation rule,
                consider each cell's left neighbor, the cell itself, and the right neighbor

                We look it up in our rule dictionary, which dictates what the cell in the
                same position, but in the next state should be.

                '001' -> '1', so the first cell becomes '1'
                '011' -> '1', so the second cell becomes '1'
                '111' -> '0', so the third cell becomes '0'
                '111' -> '0', so the fourth cell becomes '0'
                '110' -> '1', so the fifth cell becomes '1'
                '100' -> '0', so the sixth cell becomes '0'

                Therefore, the next state shall be: ['1', '1', '0', '0', '1', '0']
    """
    next_state = []

    for i in range(len(current_state)):
        left = current_state[(i - 1) % len(current_state)]
        right = current_state[(i + 1) % len(current_state)]
        curr = current_state[i]

        neighborhood = f"{left}{curr}{right}"
        if neighborhood not in automation_rule:
            raise ValueError(
                f"Given automation_rule does not contain neighborhood {neighborhood}"
            )

        next_state.append(automation_rule[neighborhood])

    return next_state
