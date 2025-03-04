# Put your openai API key here
export OAI_KEY="<API_KEY>"
export OPENAI_API_KEY="<API_KEY>"

# Put huggingface authentication token here
export HF_TOKEN="<HUGGINGFACE_AUTHENTICATION_TOKEN>"

conda activate paprika

JUDGE="gpt-4o-mini"
GAME_ENV="twenty_questions"
DATA_TYPE="train"

# Modify the path here to reflect where the paprika directory is located
cd /path/to/paprika/directory

python scripts/games/generate_twenty_questions_difficulty_levels.py judge=$JUDGE game_env=$GAME_ENV game_env.data_type=$DATA_TYPE