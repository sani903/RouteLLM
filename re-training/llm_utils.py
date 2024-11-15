import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from router_config import PROMPT_FORMAT_CONFIGS, ModelTypeEnum, RouterModelConfig
from prompt_format import PromptFormat

def load_prompt_format(model_id):
    prompt_format_dict = PROMPT_FORMAT_CONFIGS[model_id]
    return PromptFormat(**prompt_format_dict, is_generation=True)

def get_model(config: RouterModelConfig, model_ckpt: str, pad_token_id: int = 2):
    if config.model_type == ModelTypeEnum.CAUSAL:
        return AutoModelForCausalLM.from_pretrained(
            model_ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="flash_attention_2" if config.flash_attention_2 else None,
            attention_dropout=config.attention_dropout,
            token="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",
        )
    else:
        raise NotImplementedError(f"ModelType {config.model_type} is not implemented yet!")

def get_tokenizer(model_id, special_tokens=None, truncation_side="left", padding_side="left", cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        legacy=True,
        truncation_side=truncation_side,
        padding_side=padding_side,
        token="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",
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    return tokenizer

def to_openai_api_messages(system_message, classifier_message, messages):
    """Convert the conversation to OpenAI chat completion format."""

    ret = [{"role": "system", "content": system_message}]
    for i, turn in enumerate(messages):
        if i % 2 == 0:
            ret.append(
                {"role": "user", "content": classifier_message.format(question=turn)}
            )
        else:
            ret.append({"role": "assistant", "content": turn})
    return ret