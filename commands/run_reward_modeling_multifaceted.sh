#!/bin/bash

#SBATCH --job-name=mpa-Mistral-7b-v0.2-hf-rm     # Job name
#SBATCH -o /mnt/nas/suehyun/trl/logs/out_%x.txt              # Path to output log file (%j expands to job name)
#SBATCH -e /mnt/nas/suehyun/trl/logs/err_%x.err              # Path to error log file (%j expands to job name)
#SBATCH --partition=LocalQ         # Partition name
#SBATCH --nodes=1                  # Request one node
#SBATCH --ntasks=1                 # Request one task (default)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --time=24:00:00            # Time limit
#SBATCH --gres=gpu:4               # Number of GPUs to be allocated

export WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
export HF_TOKEN="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"

export WANDB_PROJECT="mpa"
export WANDB_ENTITY="suehyun"


OUTPUT_DIR="outputs/"
MODEL_NAME="kaist-ai/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
TRAIN_FILE="/mnt/nas/suehyun/MPA/data/train/preferences_v1_responses_for_reward_modeling.jsonl"
EVAL_FILE="/mnt/nas/suehyun/MPA/data/test/mpa_rm_test_set_w_rubric_per_preference.json"
EPOCHS=1
BATCH_SIZE=6
SEQ_LEN=4096
LR="5e-6"
HUB_MODEL_ID="kaist-ai/mpa-Mistral-7b-v0.2-hf-rm"


# Handle extra arguments in case one passes accelerate configs.
EXTRA_ACCELERATE_ARGS="-config_file examples/accelerate_configs/multi_gpu.yaml"
EXTRA_TRAINING_ARGS="""--gradient_accumulation_steps=4 \
  --gradient_checkpointing=True \
  --report_to="wandb" \
  --logging_steps 1 \
  --evaluation_strategy "steps" \
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 1 \
  --metric_for_best_model eval_loss \
  --load_best_model_at_end \
  --lr_scheduler_type cosine \
  --warmup_steps 10 \
  --optim adamw_bnb_8bit \
  --run_name "mpa-reward-modeling-multifaceted" \
  --push_to_hub \
  --hub_private_repo \
  --hub_strategy checkpoint
"""


srun accelerate launch $EXTRA_ACCELERATE_ARGS \
    examples/scripts/mpa/reward_modeling_multifaceted.py \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --eval_file $EVAL_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_length $SEQ_LEN \
    --learning_rate $LR \
    --hub_model_id $HUB_MODEL_ID \
    $EXTRA_TRAINING_ARGS