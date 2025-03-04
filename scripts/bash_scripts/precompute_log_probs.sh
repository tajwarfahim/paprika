# Script to precompute reference model log probs to save them in a ".pt" file
# This is done so that reference model does not need to be kept in GPU memory
# during DPO training

# activate conda environment
conda activate paprika

MODEL_MAX_LENGTHS=(20000)
SEED=69

EXPERIMENT_ID=0

# Control this based on GPU TYPE. This is required for L40S GPUs, but for some others (like A100s)
# one may want to comment out the following line
export NCCL_P2P_DISABLE=1

INFERENCE_ENGINES=(
    "llama-3.1-8b-instruct"
)

NUM_THRESHOLD_FOR_DPO=20
JUDGE_LABEL_STRATEGY="disregard_invalids"
REJECTED_SAMPLING="all"
TOKEN_LENGTH_THRESHOLD=8192


# Update the following paths depending on your setup in your local machine
# Download our preference dataset from here: https://huggingface.co/datasets/ftajwar/paprika_preference_dataset
# Convert it into a json file, put it in your local machine, 
# then put its absolute path below
DATA_DIRS=(
    /path/to/directory/containing/data/json/files
)

# Has to be a ".pt" file
SAVE_FILE_NAMES=(
    /path/to/file/that/will/hold/precomputed/log/probs/for/reference/policy.pt
)

MODEL_PATHS=(
    /path/to/reference/model
)

INFERENCE_ENGINE=${INFERENCE_ENGINES[${EXPERIMENT_ID}]}
DATA_DIR=${DATA_DIRS[${EXPERIMENT_ID}]}
SAVE_FILE_NAME=${SAVE_FILE_NAMES[${EXPERIMENT_ID}]}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTHS[${EXPERIMENT_ID}]}
MODEL_PATH=${MODEL_PATHS[${EXPERIMENT_ID}]}

# Update the directory paths below according to 
# how they are setup in your local machine
cd /path/to/paprika/codebase

python scripts/games/precompute_log_probs.py seed=$SEED rejected_sampling=$REJECTED_SAMPLING data_dir=$DATA_DIR inference_engine=$INFERENCE_ENGINE inference_engine.model_max_length=$MODEL_MAX_LENGTH save_file=$SAVE_FILE_NAME num_turn_threshold_for_dpo=$NUM_THRESHOLD_FOR_DPO inference_engine.model_name=$MODEL_PATH judge_label_strategy=$JUDGE_LABEL_STRATEGY token_length_threshold=$TOKEN_LENGTH_THRESHOLD