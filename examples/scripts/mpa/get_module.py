from trl import ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config 
import torch
from transformers import AutoModel, AutoTokenizer
import os

def get_model(model_config: ModelConfig, model_class, **kwargs):
    """
    model_config: ModelConfig
    model_class: huggingface AutoModel, e.g., AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLMWithValueHead
    """
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
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        local_files_only=True if os.path.exists(model_config.model_name_or_path) else False
    )
    model = model_class.from_pretrained(
        model_config.model_name_or_path, 
        **model_kwargs,
        **kwargs
    )
    # this was key to avoid OOM errors
    # https://github.com/huggingface/transformers/pull/24247
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

    return model

def get_tokenizer(model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    return tokenizer
    