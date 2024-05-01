from dataclasses import dataclass, field

import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, DataCollatorWithPadding, HfArgumentParser, GenerationConfig, TrainingArguments, get_cosine_schedule_with_warmup
import bitsandbytes as bnb

from torch.utils.data import DataLoader

from trl import IterativeSFTTrainer, set_seed, ModelConfig
from trl.extras import BestOfNSampler
from trl.core import LengthSampler

from get_module import get_model, get_tokenizer

tqdm.pandas()

@dataclass
class RejectionSamplingConfig:
    model_name: str = field(metadata={"help": "Name of the model."})
    reward_model: str = field(metadata={"help": "Path to the reward model."})
    query_dataset: str = field(metadata={"help": "Path to the query dataset."})
    max_length: int = field(metadata={"help": "Maximum length of the generated sequence."})
    sample_size: int = field(default=4, metadata={"help": "Number of samples to generate for each query."})

def preprocess_dataset(rs_config, tokenizer):
    raw_dataset = load_dataset('json', data_files=rs_config.query_dataset, split="train")
    original_columns = raw_dataset.column_names
    
    def apply_template(system, instruction):
        # Mistral template
        prompt = f"{system}\n{instruction}".strip()
        return f"[INST] {prompt} [/INST] "
    
    def preprocess_function(examples):
        new_examples = {
            "input_ids": []
        }
        for system, instruction in zip(examples["system"], examples["instruction"]):
            query = apply_template(system, instruction)
            tokenized_query = tokenizer(query)

            new_examples["input_ids"].append(tokenized_query["input_ids"])

        return new_examples

    preprocessed_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        remove_columns=original_columns
    )
    
    # https://github.com/huggingface/trl/issues/1473#issuecomment-2078495703
    # When the prompt exceeds the max_length, the log probabilities for both chosen and rejected turn to NaN. 
    # Consider filtering out cases where the prompt is longer than the max_length or max_prompt_len. 
    # The reason for trimming cases where the prompt exceeds max_prompt_len is that 
    # if the chosen or rejected segments are significantly shorter than the prompt, it may hinder effective learning.
    preprocessed_dataset = preprocessed_dataset.filter(
        lambda x: len(x["input_ids"]) <= rs_config.max_length
    )
    
    return preprocessed_dataset

    
def main(rs_config, training_args, model_config):
    print(os.getenv("CUDA_VISIBLE_DEVICES"))
    
    model = get_model(model_config, AutoModelForCausalLM)
    
    tokenizer = get_tokenizer(model_config)
    # https://github.com/huggingface/trl/issues/937#issuecomment-1793697802
    tokenizer.padding_side = "right"
    
    model_config.model_name_or_path = rs_config.reward_model
    reward_model = get_model(model_config, AutoModelForSequenceClassification, num_labels=1)
    
    train_dataset = preprocess_dataset(rs_config, tokenizer)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size, 
        shuffle=True, 
        collate_fn=DataCollatorWithPadding(tokenizer),
        drop_last=True
    )
    
    adam = bnb.optim.Adam8bit(model.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        adam, 
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=len(train_dataloader) * training_args.num_train_epochs
    )

    
    trainer = IterativeSFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        optimizers=(adam, lr_scheduler),
        max_length=rs_config.max_length,
        optimize_device_cache=True
    )
    
    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_config = GenerationConfig(
        min_length= -1, 
        top_k=0.0, 
        top_p= 1.0, 
        do_sample= True, 
        pad_token_id=tokenizer.eos_token_id  # for open-ended generation
    )
    
    def queries_to_scores(list_of_strings):
        # list_of_strings: candidate generated strings 
        input_ids = tokenizer(list_of_strings, max_length=rs_config.max_length, return_tensors="pt", padding=True, truncation=True).input_ids
        input_ids = input_ids.to(reward_model.device)
        reward_outputs = reward_model(input_ids)[0]
        rewards = [torch.tensor(output.item()) for output in reward_outputs]
        return rewards
        
    best_of_n = BestOfNSampler(
        model,
        tokenizer,
        queries_to_scores,
        length_sampler=LengthSampler(min_length=rs_config.max_length, max_length=rs_config.max_length),  # there's no point of using this, but BestOfNSampler requires it
        sample_size=rs_config.sample_size,
        n_candidates=1,
        generation_config=generation_config
    )

    
    for steps, batch in tqdm(enumerate(train_dataloader)):
        query_tensors = batch["input_ids"]
        output_strings = best_of_n.generate(query_tensors, device=trainer.device)
        
        step_inputs = {"texts": output_strings}
        
        trainer.step(**step_inputs)
    

if __name__ == "__main__":
    parser = HfArgumentParser((RejectionSamplingConfig, TrainingArguments, ModelConfig))
    rs_config, training_args, model_config = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)
    main(rs_config, training_args, model_config)