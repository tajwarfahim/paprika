# Put your openai API key here
export OAI_KEY="<API_KEY>"
export OPENAI_API_KEY="<API_KEY>"

# Put huggingface authentication token here
export HF_TOKEN="<HUGGINGFACE_AUTHENTICATION_TOKEN>"

conda activate paprika

AGENT="llama-3.1-8b-instruct"

MODEL_MAX_LENGTH=20000

# Pick whichever task to run inference on
# Can have values between 0 and 9
TASK_NUM=0

# Name of the task groups
GAME_ENVS=(
    "twenty_questions"
    "guess_my_city"
    "murder_mystery"
    "customer_service"
    "wordle"
    "cellular_automata"
    "mastermind"
    "battleship"
    "minesweeper"
    "bandit_bai_fixed_budget"
)

# Which task environment implementation to use
ENVS=(
    "gpt-4o-mini"
    "gpt-4o-mini"
    "gpt-4o-mini"
    "gpt-4o-mini"
    "wordle"
    "cellular_automata"
    "mastermind"
    "battleship"
    "minesweeper"
    "bandit_bai_fixed_budget"
)

# Which task judge implementation to use
JUDGES=(
    "gpt-4o-mini"
    "gpt-4o-mini"
    "gpt-4o-mini"
    "gpt-4o-mini"
    "wordle"
    "cellular_automata"
    "mastermind"
    "battleship"
    "minesweeper"
    "bandit_bai_fixed_budget"
)

GAME_ENV=${GAME_ENVS[${TASK_NUM}]}
ENV=${ENVS[${TASK_NUM}]}
JUDGE=${JUDGES}


# which split of the data to use
# DATA_TYPE="train"
DATA_TYPE="eval"

# This allows one to run evaluation only on tasks 
# belonging to indices [START_INDEX, ...., END_INDEX - 1] within the task group.
# Mostly used to split evaluation across different GPUs, feel free to modify as required.
START_INDEX=0
END_INDEX=1


# NOTE: In case one wants to run evaluation on the regular instruct model
AGENT_MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
FINETUNED_TOKENIZER=false
AGENT_SAVE_FILE_NAME="Llama-3.1-8b-instruct"

# NOTE: In case one wants to run evaluation on a trained local checkpoint
# AGENT_MODEL_NAME="/path/to/checkpoint"
# FINETUNED_TOKENIZER=true
# AGENT_SAVE_FILE_NAME="name_of_model"


# Temperature used for sampling from the agent
AGENT_TEMPERATURE=0.7
SEED=69   

# Num trajectories per task (e.g., a single secret topic in 20 questions) to be generated
NUM_TRAJECTORIES_PER_GAME_SCENARIO=4

# Modify the path here to reflect where the paprika directory is located
cd /path/to/paprika/directory
python scripts/games/run_data_generation_game.py seed=$SEED agent=$AGENT env=$ENV judge=$JUDGE game_env=$GAME_ENV game_env.data_type=$DATA_TYPE start_index=$START_INDEX end_index=$END_INDEX agent.model_name=$AGENT_MODEL_NAME agent.tokenizer_name=$AGENT_MODEL_NAME agent.finetuned_tokenizer=$FINETUNED_TOKENIZER agent.save_file_name=$AGENT_SAVE_FILE_NAME agent.model_max_length=$MODEL_MAX_LENGTH agent_temperature=$AGENT_TEMPERATURE num_trajectories_per_game_scenario=$NUM_TRAJECTORIES_PER_GAME_SCENARIO agent_model_supports_system_message=true