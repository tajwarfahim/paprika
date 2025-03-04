from llm_exploration.inference.inference_engine import LLMInferenceEngine

from llm_exploration.inference.huggingface_inference_engine import (
    HuggingFaceLLMInferenceEngine,
)

from llm_exploration.inference.openai_inference_engine import OpenAIInferenceEngine

from llm_exploration.inference.wordle_engine import WordleInferenceEngine

from llm_exploration.inference.wordle_modified_engine import WordleModifiedInferenceEngine

from llm_exploration.inference.elementary_cellular_automation_inference_engine import (
    CellularAutomationInferenceEngine,
)

from llm_exploration.inference.bandit_engine import BanditInferenceEngine

from llm_exploration.inference.bandit_engine_bai_from_fixed_sampling_budget import (
    BanditBAIFixedBudgetInferenceEngine,
)

from llm_exploration.inference.mastermind_engine import MastermindInferenceEngine

from llm_exploration.inference.minesweeper_engine import (
    MinesweeperInferenceEngine,
    MinesweeperJudgeInferenceEngine,
)

from llm_exploration.inference.battleship_engine import BattleshipInferenceEngine

try:
    from llm_exploration.inference.jericho_inference_engine import JerichoInferenceEngine
except:
    print("Could not import JerichoInferenceEngine, so cannot use it!")

try:
    from llm_exploration.inference.vllm_inference_engine import VLLMInferenceEngine
except:
    print("Could not import VLLMInferenceEngine, so cannot use it!")
