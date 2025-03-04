import os

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PARENT_DIR, "data")
SCRIPTS_DIR = os.path.join(PARENT_DIR, "scripts")

EXPERIMENT_SCRIPTS_DIR = os.path.join(SCRIPTS_DIR, "experiments")
EXPERIMENT_LOG_DIR = os.path.join(SCRIPTS_DIR, "logs")
EXPERIMENT_CONFIG_DIR = os.path.join(EXPERIMENT_SCRIPTS_DIR, "configs")
DATAGEN_CONFIG_DIR = os.path.join(SCRIPTS_DIR, "configs")
GAME_CONFIFS_DIR = os.path.join(PARENT_DIR, "llm_exploration", "game", "game_configs")

JERICHO_GAMES_DIR = os.path.join(PARENT_DIR, "jericho_games")

WANDB_DIR = EXPERIMENT_LOG_DIR
