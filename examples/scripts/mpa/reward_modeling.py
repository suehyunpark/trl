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
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_quantization_config

tqdm.pandas()




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
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )
        
    return model, tokenizer


def apply_template_mistral_instruct(system, instruction, response):
    prompt = f"{system}\n{instruction}".strip()
    return f"[INST] {prompt} [/INST] {response}"


def preprocess_dataset(dataset_path, tokenizer):
    raw_datasets = load_dataset(dataset_path)
    
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        system = examples["system"]
        instruction = examples["instruction"]
        output = examples["output"]
        for chosen, rejected in zip(output["chosen"], output["rejected"]):
            tokenized_chosen = tokenizer(
                apply_template_mistral_instruct(system, instruction, chosen), 
                truncation=True, 
                max_length=reward_config.max_length
            )
            tokenized_rejected = tokenizer(
                apply_template_mistral_instruct(system, instruction, rejected), 
                truncation=True, 
                max_length=reward_config.max_length
            )

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    return raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    

def main(reward_config, model_config, args):
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    model, tokenizer = get_model_tokenizer(model_config)
    train_dataset = preprocess_dataset(args.train_file, tokenizer)
    eval_dataset = preprocess_dataset(args.eval_file, tokenizer)
    
    
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
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    reward_config, model_config, args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    main(reward_config, model_config, args)