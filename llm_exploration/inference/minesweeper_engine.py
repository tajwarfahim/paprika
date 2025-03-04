import time
import re
import random
from collections import deque
from typing import List, Dict, Optional, Tuple
from llm_exploration.inference.inference_engine import LLMInferenceEngine


class MinesweeperInferenceEngine(LLMInferenceEngine):
    """
    Inspired by: https://en.wikipedia.org/wiki/Minesweeper_(video_game)
    made into a text-based Minesweeper environment. It interprets user moves like:
    "reveal 2 3"
    which updates the board and returns feedback.
    If there is a mine in the position, the game will be over.
    If there are mines in adjacent cells, the number will be revealed.
    If there are 0 mines in adjacent cells, there will be a BFS flood fill reveal.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.rows = 0
        self.cols = 0
        self.mine_positions = set()
        self.revealed = set()
        self.game_over = False
        self.num_safe_cells = 0
        self._initialized = False
        self.adjacent_counts = {}
        self._random_mine_count = 0

    def soft_reset(self) -> None:
        self.revealed = set()
        self.game_over = False
        self.adjacent_counts = {}

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        """
        Main method for interpreting a single user move:
         1. If not initialized, parse system prompt for board size & possible mine positions.
         2. Parse user's last message for "reveal <row> <col>".
         3. On the first reveal, if no mines are predefined, do random placement excluding that cell.
         4. Reveal the cell:
             - If it's a mine -> game over.
             - Otherwise, store adjacency count. If 0, BFS flood fill.
         5. Check win condition. Return partial board or messages accordingly.
        """
        time.sleep(0.1)
        self.validate_conversation(conv)

        if not self._initialized:
            self._parse_board_from_system(conv[0]["content"])
            self._initialized = True

        user_message = conv[-1]["content"].strip().lower()

        row, col = self._parse_reveal_command(user_message)
        if row is None or col is None:
            return "Invalid move format. Use: 'reveal <row> <col>'. " "Example: 'reveal 1 3'."

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return (
                f"Invalid move: ({row},{col}) is out of bounds "
                f"(valid range: rows between 0 and {self.rows-1}, "
                f"cols betweewn 0 and {self.cols-1})"
            )

        if (row, col) in self.revealed:
            return f"Cell ({row},{col}) is already revealed."

        if len(self.mine_positions) == 0 and len(self.revealed) == 0:
            self._place_mines_excluding(row, col, self._random_mine_count)

        self.revealed.add((row, col))

        if (row, col) in self.mine_positions:
            self.game_over = True
            return f"BOOM! You hit a mine at ({row},{col}). <End> Game over </End>"

        adj_mines = self._count_adj_mines(row, col)
        self.adjacent_counts[(row, col)] = adj_mines

        if adj_mines == 0:
            self._flood_fill(row, col)

        if self._check_victory():
            self.game_over = True
            return "Goal reached"

        return self._render_board_message()

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        self.validate_batch_of_conversations(convs)
        return [self.generate(conv, max_n_tokens, temperature, top_p, min_p) for conv in convs]

    def _parse_board_from_system(self, system_prompt: str) -> None:
        """
        parse rows and cols.
        """
        lines = system_prompt.strip().split("\n")
        if len(lines) < 1:
            raise ValueError("System prompt must have at least one line: 'rows cols'.")

        first_line = lines[0].strip()
        parts = first_line.split()
        if len(parts) != 2:
            raise ValueError("First line must have exactly two integers, e.g. '5 5'.")
        self.rows = int(parts[0])
        self.cols = int(parts[1])
        suggested = max(1, (self.rows * self.cols) // 10)
        self._random_mine_count = min(suggested, self.rows * self.cols - 1)

    def _parse_reveal_command(self, msg: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Only accepts "reveal <row> <col>".
        Returns (row, col) or (None, None) if parsing fails.
        """
        pattern = r"^reveal\s+(\d+)\s+(\d+)$"
        match = re.search(pattern, msg)
        if match:
            return int(match.group(1)), int(match.group(2))
        return (None, None)

    def _place_mines_excluding(self, safe_r: int, safe_c: int, total_mines: int) -> None:
        """
        Randomly place `total_mines` mines in the grid, ensuring the first reveal
        cells are safe.
        """
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        all_cells.remove((safe_r, safe_c))
        mine_cells = random.sample(all_cells, min(total_mines, len(all_cells)))
        for cell in mine_cells:
            self.mine_positions.add(cell)
        self.num_safe_cells = (self.rows * self.cols) - len(self.mine_positions)

    def _count_adj_mines(self, row: int, col: int) -> int:
        """
        Count how many mines are adjacent to (row, col).
        """
        count = 0
        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if (r, c) in self.mine_positions:
                        count += 1
        return count

    def _flood_fill(self, start_r: int, start_c: int) -> None:
        """
        flood fill for any revealed cell that has 0 adjacent mines.
        Expands to neighbors that are not mines and not revealed yet.
        """
        queue = deque()
        queue.append((start_r, start_c))

        while queue:
            r, c = queue.popleft()

            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        if (rr, cc) not in self.mine_positions and (
                            rr,
                            cc,
                        ) not in self.revealed:
                            self.revealed.add((rr, cc))
                            adj_count = self._count_adj_mines(rr, cc)
                            self.adjacent_counts[(rr, cc)] = adj_count
                            if adj_count == 0:
                                queue.append((rr, cc))

    def _check_victory(self) -> bool:
        """
        Check if all safe cells have been revealed.
        """
        return len(self.revealed) == self.num_safe_cells

    def _render_board_message(self) -> str:
        """
        Returns a text-based partial board for debugging/feedback.
        If adjacency == 0, display '*'.
        Else display the adjacency count.
        Hidden cells remain '#' until the game is over (win or lose).
        Once game_over is True, display all mines as 'M'.
        """
        lines = []
        reveal_mines_now = self.game_over

        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                if (r, c) in self.revealed and (r, c) not in self.mine_positions:
                    count = self.adjacent_counts.get((r, c), 0)
                    row_chars.append(str(count) if count > 0 else "*")
                elif reveal_mines_now and (r, c) in self.mine_positions:
                    row_chars.append("M")
                else:
                    row_chars.append("#")
            lines.append(" ".join(row_chars))

        return "\n".join(lines)


class MinesweeperJudgeInferenceEngine(LLMInferenceEngine):
    """
    This engine checks if the agent has successfully solved the game,
    or the game ended because of failure.
    """

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        env_last_message = conv[-1]["content"]

        if "goal reached" in env_last_message.lower():
            return "<VALID>"

        else:
            return "<NOTVALID>"

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        self.validate_batch_of_conversations(convs)
        return [self.generate(conv, max_n_tokens, temperature, top_p, min_p) for conv in convs]
