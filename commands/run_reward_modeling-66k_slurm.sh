#!/bin/bash

#SBATCH --job-name=mpa-Mistral-7b-v0.2-hf-rm-66k     # Job name
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

OUTPUT_DIR="outputs/"
MODEL_NAME="/mnt/nas/suehyun/axolotl/outputs/mpa/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
TRAIN_FILE="/mnt/nas/suehyun/MPA/data/train/preferences_v1_responses_for_orpo_64k_v2_ppo.jsonl"
EVAL_FILE="/mnt/nas/suehyun/MPA/data/test/mpa_rm_test_set_w_rubric_per_preference.json"
EPOCHS=1
BATCH_SIZE=4
SEQ_LEN=4096
# LR="2e-5"
LR="5e-6"

HUB_MODEL_ID="kaist-ai/mpa-Mistral-7b-v0.2-hf-rm-66k"
RUN_NAME="mpa-Mistral-7b-v0.2-hf-rm-66k"


# Handle extra arguments in case one passes accelerate configs.
# EXTRA_ACCELERATE_ARGS="--config_file examples/accelerate_configs/multi_gpu_slurm.yaml"
EXTRA_ACCELERATE_ARGS="--config_file examples/accelerate_configs/deepspeed_zero2.yaml"

MODEL_CONFIG_ARGS="""--model_name_or_path $MODEL_NAME \
  --torch_dtype float16
"""
#   --load_in_8bit=True \
#   --use_peft=True \
#   --lora_task_type="SEQ_CLS" \
#   --lora_r 8 \
#   --lora_alpha 32 \
#   --lora_dropout 0.1
# """

REWARD_CONFIG_ARGS="""--output_dir $OUTPUT_DIR \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --max_length $SEQ_LEN \
  --learning_rate $LR \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing=True \
  --logging_steps 1 \
  --evaluation_strategy "steps" \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 1 \
  --metric_for_best_model accuracy \
  --load_best_model_at_end \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --warmup_steps 10 \
  --report_to="wandb" \
  --push_to_hub \
  --hub_model_id $HUB_MODEL_ID \
  --hub_private_repo \
  --hub_strategy checkpoint \
  --run_name $RUN_NAME \
  --fp16=True"""


srun accelerate launch $EXTRA_ACCELERATE_ARGS \
    examples/scripts/mpa/reward_modeling.py \
    --train_file $TRAIN_FILE \
    --eval_file $EVAL_FILE \
    $MODEL_CONFIG_ARGS \
    $REWARD_CONFIG_ARGS
    