import re
import random
import time
from copy import deepcopy
from typing import List, Dict, Optional

from llm_exploration.inference.inference_engine import LLMInferenceEngine

# from inference_engine import LLMInferenceEngine

MAX_GUESS = 15

SHIP_SIZES = {"Carrier": 5, "Battleship": 4, "Destroyer": 2}


class BattleshipInferenceEngine(LLMInferenceEngine):
    """
    Inspired by: https://en.wikipedia.org/wiki/Battleship_(game)
    Slightly modfied due to making it a single-player Battleship engine.
    The grid size is variable and fixed by system prompt and three types of ship
    instead of traditional 5. Since the assistant will always win as there is no
    competitor to win before the number of guesses has been limited 15,
    it will give it free guesses and number of free guesses will vary
    based on the board size.
    """

    def __init__(
        self,
    ):
        print("Battleship engine started!")

    def soft_reset(self) -> None:
        self.ship_cells_hit = 0
        self.guesses_made = 0

        if self.grid_size is not None:
            self.board = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            self.ship_positions = deepcopy(self.ship_positions_at_the_start)

            for _, positions in self.ship_positions.items():
                for row_index, col_index in positions:
                    self.board[row_index][col_index] = "S"

        super().soft_reset()

    def reset(self) -> None:
        self.board = None
        self.grid_size = None
        self.ship_positions = None
        self.ship_positions_at_the_start = None

    def setup_game_internally(self, grid_size: int) -> None:
        if self.grid_size is None:
            self.grid_size = grid_size
            self.ship_sizes = SHIP_SIZES

            self.board = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            self.ship_positions = {ship_name: set() for ship_name in self.ship_sizes}

            # Randomly place ships (internally stored as 'S')
            self._place_ships_randomly()
            self.ship_positions_at_the_start = deepcopy(self.ship_positions)

            self.total_ship_cells = sum(self.ship_sizes.values())  # 5 + 4 + 2 = 11
            self.ship_cells_hit = 0

            self.guesses_made = 0

    def _place_ships_randomly(self):
        for ship_name, size in self.ship_sizes.items():
            placed = False
            while not placed:
                orientation = random.choice(["horizontal", "vertical"])
                row = random.randint(0, self.grid_size - 1)
                col = random.randint(0, self.grid_size - 1)

                if self._can_place_ship(row, col, size, orientation):
                    self._do_place_ship(ship_name, row, col, size, orientation)
                    placed = True

    def _can_place_ship(self, row, col, size, orientation) -> bool:
        if orientation == "horizontal":
            if col + size > self.grid_size:
                return False
            for c in range(col, col + size):
                if self.board[row][c] != ".":
                    return False
        else:
            if row + size > self.grid_size:
                return False
            for r in range(row, row + size):
                if self.board[r][col] != ".":
                    return False
        return True

    def _do_place_ship(
        self, ship_name: str, row: int, col: int, size: int, orientation: str
    ) -> None:
        if orientation == "horizontal":
            for c in range(col, col + size):
                self.board[row][c] = "S"
                self.ship_positions[ship_name].add((row, c))
        else:
            for r in range(row, row + size):
                self.board[r][col] = "S"
                self.ship_positions[ship_name].add((r, col))

    def _board_to_string(self) -> str:
        header_nums = " ".join(str(i + 1).rjust(2) for i in range(self.grid_size))
        header = f"    {header_nums}"

        rows = []
        for r in range(self.grid_size):
            row_label = chr(ord("A") + r)
            row_cells = []

            for c in range(self.grid_size):
                val = self.board[r][c]

                if val == "S":
                    row_cells.append(".")

                else:
                    row_cells.append(val)

            row_str = " ".join(cell.rjust(2) for cell in row_cells)
            rows.append(f"{row_label}  {row_str}")
        return header + "\n" + "\n".join(rows)

    def _board_to_string_internal(self) -> str:
        header_nums = " ".join(str(i + 1).rjust(2) for i in range(self.grid_size))
        header = f"    {header_nums}"

        rows = []
        for r in range(self.grid_size):
            row_label = chr(ord("A") + r)
            row_cells = []

            for c in range(self.grid_size):
                val = self.board[r][c]
                row_cells.append(val)

            row_str = " ".join(cell.rjust(2) for cell in row_cells)
            rows.append(f"{row_label}  {row_str}")
        return header + "\n" + "\n".join(rows)

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        """
        Processes the latest guess from the user (the last message in conv).
        Prints the updated board from inside this method.
        Returns the feedback for that guess.

        In the board, '.'-> unreaveled, 'M'->Missed, 'H'->Hit
        """
        if self.guesses_made == 0:
            system_prompt = conv[0].get("content", "")
            from_prompt = extract_grid_size_from_system_prompt(system_prompt, default_size=5)
            self.setup_game_internally(grid_size=from_prompt)

        time.sleep(0.1)

        guess_text = conv[-1]["content"].strip().upper()

        pattern = r"^([A-Z])(\d{1,2})$"
        match = re.match(pattern, guess_text)
        if not match:
            feedback = f"Invalid guess '{guess_text}'. " "Please guess e.g. A1, B3, etc."
            return feedback

        row_letter = match.group(1)
        col_number = int(match.group(2))

        row_index = ord(row_letter) - ord("A")
        if not (0 <= row_index < self.grid_size):
            max_row_letter = chr(ord("A") + self.grid_size - 1)
            feedback = (
                f"Row '{row_letter}' is out of range. Valid rows are A..{max_row_letter}."
            )
            return feedback

        if not (1 <= col_number <= self.grid_size):
            feedback = f"Column {col_number} out of range. Must be 1..{self.grid_size}."
            return feedback

        col_index = col_number - 1

        self.guesses_made += 1
        cell_value = self.board[row_index][col_index]

        if cell_value == "S":
            self.board[row_index][col_index] = "X"
            self.ship_cells_hit += 1

            just_sank_ship = False
            sunk_ship_name = None
            hit_ship_name = None

            for ship_name, positions in self.ship_positions.items():
                if (row_index, col_index) in positions:
                    positions.remove((row_index, col_index))
                    hit_ship_name = ship_name
                    if not positions:
                        just_sank_ship = True
                        sunk_ship_name = ship_name
                    break

            if self.ship_cells_hit == self.total_ship_cells:
                feedback = (
                    f"Hit at {guess_text}! You sank the {sunk_ship_name}!\n"
                    f"You have sunk all ships in {self.guesses_made} guesses. "
                    f"You Won! Goal reached."
                )

            else:
                feedback = f"Hit at {guess_text}!"
                if just_sank_ship:
                    feedback += f" You sank the {sunk_ship_name}!"
                else:
                    feedback += (
                        f" You have hit a {hit_ship_name}, which occupies "
                        f"{SHIP_SIZES[hit_ship_name]} cells in the grid.\n"
                    )

                feedback += " Here is how the board looks now: \n" + self._board_to_string()

        elif cell_value in ("X", "M"):
            feedback = f"You already guessed {guess_text}. Please make another guess!"

            feedback += " Here is how the board looks now: \n" + self._board_to_string()

        else:
            self.board[row_index][col_index] = "M"

            feedback = f"Miss at {guess_text}. There is no ship in this co-ordinate."

            feedback += " Here is how the board looks now: \n" + self._board_to_string()

        return feedback

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        feedbacks = []
        for conv in convs:
            feedback = self.generate(
                conv=conv,
                max_n_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
            )
            feedbacks.append(feedback)
        return feedbacks


def extract_grid_size_from_system_prompt(system_prompt: str, default_size: int = 5) -> int:
    match = re.search(r"grid\s*size\s*:\s*(\d+)", system_prompt, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default_size
