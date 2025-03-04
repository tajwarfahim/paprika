conda activate paprika

# Control this based on GPU TYPE. This is required for L40S GPUs, but for some others (like A100s)
# one may want to comment out the following line
export NCCL_P2P_DISABLE=1

SEED=69
NUM_THRESHOLD_FOR_DPO=20
JUDGE_LABEL_STRATEGY="disregard_invalids"

# Point to huggingface model name OR path to a local trained checkpoint
# of the model that one wants to finetune
MODEL_PATHS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # /path/to/local/model/checkpoint
)

TOKENIZER_NAMES=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# control the name of the saved checkpoint
MODEL_NAMES=(
    LLama-3.1-8B
)


# Put the local path to precomputed log probs from the reference model + trajectories

# NOTE: This has to be a ".pt" file
TRAIN_DATA_PATHS=(
    /path/to/local/dataset
)

# If you also have an evaluation dataset, containing precomputed log probs + trajectories
# put the path to it here

# NOTE: This has to be a ".pt" file
EVAL_DATA_PATHS=(
    /path/to/local/evaluation/dataset
)

MAX_LENGTHS=(131072)

MODEL_PATH=${MODEL_PATHS[${SLURM_ARRAY_TASK_ID}]}
MODEL_NAME=${MODEL_NAMES[${SLURM_ARRAY_TASK_ID}]}
TOKENIZER_NAME=${TOKENIZER_NAMES[${SLURM_ARRAY_TASK_ID}]}

TRAIN_DATA_DIR=${TRAIN_DATA_DIRS[${SLURM_ARRAY_TASK_ID}]}
EVAL_DATA_DIR=${EVAL_DATA_DIRS[${SLURM_ARRAY_TASK_ID}]}

TRAIN_DATA_PATH=${TRAIN_DATA_PATHS[${SLURM_ARRAY_TASK_ID}]}
EVAL_DATA_PATH=${EVAL_DATA_PATHS[${SLURM_ARRAY_TASK_ID}]}

MAX_LENGTH=${MAX_LENGTHS[${SLURM_ARRAY_TASK_ID}]}

ADD_ANSWER_TOKEN=false
MASTER_PORT=29600

TRAINER_TYPE="DPO"
LEARNING_RATE=2e-7

# Hyperparams for other trainers
GEM_BETA=0.7
GEM_H_FUNCTION="linear"

# beta parameter for DPO
DPO_BETA=0.1

REJECTED_TRAJECTORY_SAMPLING_STRATEGY="all"

# RPO alpha parameter, controlling the weight of the SFT loss
# Total loss = DPO loss (preferred, dispreferred) + RPO_ALPHA * SFT loss (preferred)
RPO_ALPHA=1.0

SIMPO_BETA=2.5
SIMPO_GAMMA=1.4

DPO_REGULARIZATION_COEFFICIENT=1.0

TOKEN_LENGTH_THRESHOLD=8192

# Update the directory paths below according to 
# how they are setup in your local machine
RUN_NAME="${MODEL_NAME}_${TRAINER_TYPE}_${LEARNING_RATE}";
LOGDIR="/path/to/logdir/$RUN_NAME";
REPO_DIR="/path/to/paprika/codebase";

deepspeed --master_port $MASTER_PORT $REPO_DIR/scripts/games/run_llm_finetuning_on_game_data.py  \
        --model_name_or_path $MODEL_PATH \
        --seed $SEED  \
        --train_data_dir null \
        --eval_data_dir null \
        --train_dataset_path_dpo $TRAIN_DATA_PATH \
        --eval_dataset_path_dpo $EVAL_DATA_PATH \
        --output_dir $LOGDIR  \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3.json  \
        --bf16 True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1   \
        --gradient_accumulation_steps 4 \
        --eval_strategy "steps" \
        --eval_steps 10000  \
        --save_strategy "steps"  \
        --save_steps 10000  \
        --save_total_limit 1 \
        --learning_rate $LEARNING_RATE  \
        --weight_decay 0.    \
        --warmup_ratio 0.04   \
        --lr_scheduler_type "cosine"   \
        --logging_steps 1     \
        --tf32 True    \
        --model_max_length $MAX_LENGTH \
        --gradient_checkpointing True \
        --remove_unused_columns False \
        --tokenizer_name $TOKENIZER_NAME \
        --add_answer_tokens $ADD_ANSWER_TOKEN \
        --trainer_type $TRAINER_TYPE \
        --gem_beta $GEM_BETA \
        --gem_h_function $GEM_H_FUNCTION \
        --dpo_beta $DPO_BETA \
        --rejected_trajectory_sampling_strategy $REJECTED_TRAJECTORY_SAMPLING_STRATEGY \
        --wandb_project "Paprika" \
        --simpo_beta $SIMPO_BETA \
        --simpo_gamma $SIMPO_GAMMA \
        --rpo_alpha $RPO_ALPHA \
        --regularization_coefficient $DPO_REGULARIZATION_COEFFICIENT \
	    --num_turn_threshold_for_dpo $NUM_THRESHOLD_FOR_DPO \
        --judge_label_strategy $JUDGE_LABEL_STRATEGY \
        --token_length_threshold $TOKEN_LENGTH_THRESHOLD \
        --copy_instruct_tokenizer_chat_template False \
        --should_resume_training_from_checkpoint False \
