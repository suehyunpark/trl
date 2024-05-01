#!/bin/bash

#SBATCH --job-name=mpa-Mistral-7b-v0.2-hf-ppo-66k     # Job name
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

export WANDB_PROJECT="mpa-rm"
export WANDB_ENTITY="suehyun"

# https://github.com/ultralytics/ultralytics/issues/1439#issuecomment-1821973338
export NCCL_IB_GID_INDEX=3
# https://github.com/huggingface/accelerate/issues/314#issuecomment-1565259831
export NCCL_P2P_LEVEL=NVL

OUTPUT_DIR="outputs/mpa-Mistral-7b-v0.2-hf-ppo-66k"
MODEL_NAME="/mnt/nas/suehyun/axolotl/outputs/mpa/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
REWARD_MODEL_NAME="kaist-ai/mpa-Mistral-7b-v0.2-hf-rm-66k"
TRAIN_FILE="/mnt/nas/suehyun/MPA/data/train/preferences_v1_responses_for_orpo_64k_v2_ppo.jsonl"
EVAL_FILE="/mnt/nas/suehyun/MPA/data/test/mpa_rm_test_set_w_rubric_per_preference.json"
EPOCHS=3
BATCH_SIZE=4
SEQ_LEN=4096
# LR="2e-5"
LR="1.41e-5"  # InstructGPT

HUB_MODEL_ID="kaist-ai/mpa-Mistral-7b-v0.2-hf-ppo-66k"
RUN_NAME="mpa-Mistral-7b-v0.2-hf-ppo-66k"


# Handle extra arguments in case one passes accelerate configs.
# EXTRA_ACCELERATE_ARGS="--config_file examples/accelerate_configs/multi_gpu_slurm.yaml"
EXTRA_ACCELERATE_ARGS="--config_file examples/accelerate_configs/deepspeed_zero2.yaml"

DATA_CONFIG_ARGS = """--max_length $SEQ_LEN \
  --repo_id $HUB_MODEL_ID \
  --run_name $RUN_NAME \
  --output_dir $OUTPUT_DIR \
  --save_steps 500
"""

PPO_CONFIG_ARGS="""--exp_name $RUN_NAME \
  --seed 42 \
  --log_with wandb \
  --model_name $MODEL_NAME \
  --reward_model $REWARD_MODEL_NAME \
  --query_dataset $TRAIN_FILE \
  --remove_unused_columns True \
  --learning_rate $LR \
  --batch_size $BATCH_SIZE \
  --mini_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps 1 \
  --ppo_epochs $EPOCHS \
  --optimize_device_cache True \
  --tracker_project_name $WANDB_PROJECT \
  --trust_remote_code True"""


srun accelerate launch $EXTRA_ACCELERATE_ARGS \
    examples/scripts/mpa/ppo.py \
    $DATA_CONFIG_ARGS \
    $PPO_CONFIG_ARGS
    