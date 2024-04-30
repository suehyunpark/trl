#!/bin/bash

export WANDB_API_KEY="339cad8697ca8b7558010d3f8c4aa40788e64d12"
export HF_TOKEN="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"

export WANDB_PROJECT="mpa-rm"
export WANDB_ENTITY="suehyun"

export NCCL_IB_GID_INDEX=3
export NCCL_P2P_LEVEL=NVL

OUTPUT_DIR="outputs/"
MODEL_NAME="kaist-ai/mpa-Mistral-7b-v0.2-hf-sft-epoch1"
TRAIN_FILE="/data/suehyun/MPA/data/train/preferences_v1_responses_for_orpo_64k_v2_ppo.jsonl"
EVAL_FILE="/data/suehyun/MPA/data/test/mpa_rm_test_set_w_rubric_per_preference.json"
EPOCHS=1
BATCH_SIZE=4
SEQ_LEN=4096
# LR="2e-5"
LR="9e-6"

HUB_MODEL_ID="kaist-ai/mpa-Mistral-7b-v0.2-hf-rm-66k"
RUN_NAME="mpa-Mistral-7b-v0.2-hf-rm-66k"


# Handle extra arguments in case one passes accelerate configs.
# EXTRA_ACCELERATE_ARGS="--config_file examples/accelerate_configs/multi_gpu_slurm.yaml"
EXTRA_ACCELERATE_ARGS="--config_file examples/accelerate_configs/deepspeed_zero2.yaml"
CUDA_VISIBLE_DEVICES="0,1,2,3"

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
  --eval_steps 50 \
  --save_steps 100 \
  --save_total_limit 1 \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --warmup_steps 10 \
  --report_to="wandb" \
  --push_to_hub \
  --hub_model_id $HUB_MODEL_ID \
  --hub_private_repo \
  --hub_strategy checkpoint \
  --run_name $RUN_NAME \
  --fp16=True \
  --seed 42"""


CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch $EXTRA_ACCELERATE_ARGS \
    examples/scripts/mpa/reward_modeling.py \
    --train_file $TRAIN_FILE \
    --eval_file $EVAL_FILE \
    $MODEL_CONFIG_ARGS \
    $REWARD_CONFIG_ARGS
    
