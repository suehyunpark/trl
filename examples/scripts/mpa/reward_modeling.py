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
from typing import Dict
import numpy as np

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_quantization_config

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


def get_model_tokenizer(model_config):
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, 
        num_labels=1, 
        attn_implementation="flash_attention_2",
        **model_kwargs
    )
    # this was key to avoid OOM errors
    # https://github.com/huggingface/transformers/pull/24247
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )
        
    return model, tokenizer


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
        num_proc=8,
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
    
# log error, 다른 거에 wrapping
def main(reward_config, model_config, data_config):
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    # print(os.getenv("CUDA_VISIBLE_DEVICES"))

    model, tokenizer = get_model_tokenizer(model_config)
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
        eval_dataset=eval_dataset
    )
    trainer.train()
    trainer.save_model(reward_config.output_dir)



if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig, DataConfig))
    reward_config, model_config, data_config = parser.parse_args_into_dataclasses()
    
    main(reward_config, model_config, data_config)