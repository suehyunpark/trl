# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import torch
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field
import numpy as np
from typing import Dict
from scipy import stats
import traceback
import sys
import wandb

from trl import ModelConfig, RewardConfig, RewardTrainer, set_seed, get_kbit_device_map, get_quantization_config
from get_module import get_model, get_tokenizer

tqdm.pandas()

DEBUG = os.getenv("DEBUG", False)

@dataclass
class DataConfig:
    train_file: str = field(
        metadata={"help": "Path to the training dataset file."}
    )
    eval_file: str = field(
        metadata={"help": "Path to the evaluation dataset file."}
    )
    
# trl/trainer/utils.py#L548-L549
def compute_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    if np.array(predictions[:, 0] == predictions[:, 1], dtype=float).sum() > 0:
        warnings.warn(
            f"There are {np.array(predictions[:, 0] == predictions[:, 1]).sum()} out of {len(predictions[:, 0])} instances where the predictions for both options are equal. As a consequence the accuracy can be misleading."
        )
    rewards_chosen_stats = stats.describe(predictions[:, 0])
    rewards_rejected_stats = stats.describe(predictions[:, 1])
    
    rewards_chosen_mean = rewards_chosen_stats.mean
    rewards_rejected_mean = rewards_rejected_stats.mean
    rewards_chosen_std = rewards_chosen_stats.variance ** 0.5
    rewards_rejected_std = rewards_rejected_stats.variance ** 0.5

    predictions = np.argmax(predictions, axis=1)
    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    
    return {
        "accuracy": accuracy, 
        "rewards_chosen_mean": rewards_chosen_mean, 
        "rewards_rejected_mean": rewards_rejected_mean,
        "rewards_chosen_std": rewards_chosen_std,
        "rewards_rejected_std": rewards_rejected_std
    }


# def get_model_tokenizer(model_config):
#     torch_dtype = (
#         model_config.torch_dtype
#         if model_config.torch_dtype in ["auto", None]
#         else getattr(torch, model_config.torch_dtype)
#     )
#     quantization_config = get_quantization_config(model_config)
#     model_kwargs = dict(
#         revision=model_config.model_revision,
#         trust_remote_code=model_config.trust_remote_code,
#         device_map=get_kbit_device_map() if quantization_config is not None else None,
#         quantization_config=quantization_config,
#         attn_implementation=model_config.attn_implementation,
#         torch_dtype=torch_dtype
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_config.model_name_or_path, 
#         num_labels=1, 
#         **model_kwargs
#     )
#     # this was key to avoid OOM errors
#     # https://github.com/huggingface/transformers/pull/24247
#     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

#     if model_config.lora_task_type != "SEQ_CLS":
#         warnings.warn(
#             "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
#             " Make sure to pass --lora_task_type SEQ_CLS when using this script."
#         )
        
#     return model, tokenizer


def preprocess_dataset(dataset_path, reward_config, tokenizer):
    raw_datasets = load_dataset('json', data_files=dataset_path)
    
    def apply_template(system, instruction, response):
        # Mistral template
        prompt = f"{system}\n{instruction}".strip()
        return f"[INST] {prompt} [/INST] {response}"
    
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for system, instruction, chosen, rejected in zip(examples["system"], examples["instruction"], examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(
                apply_template(system, instruction, chosen), 
                # truncation=True, 
                # max_length=reward_config.max_length
            )
            tokenized_rejected = tokenizer(
                apply_template(system, instruction, rejected), 
                # truncation=True, 
                # max_length=reward_config.max_length
            )

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    preprocessed_dataset = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=24,
    )["train"]
    
    # https://github.com/huggingface/trl/issues/1473#issuecomment-2078495703
    # When the prompt exceeds the max_length, the log probabilities for both chosen and rejected turn to NaN. 
    # Consider filtering out cases where the prompt is longer than the max_length or max_prompt_len. 
    # The reason for trimming cases where the prompt exceeds max_prompt_len is that 
    # if the chosen or rejected segments are significantly shorter than the prompt, it may hinder effective learning.
    preprocessed_dataset = preprocessed_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
    
    return preprocessed_dataset
    
def main(reward_config, model_config, data_config):
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    print(os.getenv("CUDA_VISIBLE_DEVICES"))

    model = get_model(model_config, AutoModelForSequenceClassification, num_labels=1)
    tokenizer = get_tokenizer(model_config)
    # model, tokenizer = get_model_tokenizer(model_config)
    
    # https://github.com/huggingface/trl/issues/937#issuecomment-1793697802
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    
    train_dataset = preprocess_dataset(data_config.train_file, reward_config, tokenizer)
    eval_dataset = preprocess_dataset(data_config.eval_file, reward_config, tokenizer)
    
    # print("Number of parameters in the model:")
    # print(model.num_parameters())
    # print("Number of trainable parameters in the model:")
    # print(model.num_parameters(only_trainable=True))
        
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    try:
        trainer.train()
        trainer.save_model(reward_config.output_dir)
    except Exception as e:
        print(traceback.print_exc(), file=sys.stderr)


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig, DataConfig))
    reward_config, model_config, data_config = parser.parse_args_into_dataclasses()
    
    set_seed(reward_config.seed)
    main(reward_config, model_config, data_config)