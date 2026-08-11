"""
Everything specific to Qwen lives here. Qwen's chat template supports a
system role natively, so no merging is needed -- otherwise identical
mechanics to every other model.

NOTE: if pointing this at a "thinking"-capable Qwen3 checkpoint, increase
max_new_tokens substantially, since it may emit reasoning text before
its actual answer. Using a non-thinking checkpoint (e.g. an
"-Instruct-2507" release) avoids this entirely.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_path, torch_dtype="bfloat16", device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, torch_dtype), device_map=device_map
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_one(model, tokenizer, messages, max_new_tokens=10):
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def generate_batch(model, tokenizer, messages_list, max_new_tokens=10):
    tokenizer.padding_side = "left"
    prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages_list]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(output[i][prompt_len:], skip_special_tokens=True).strip() for i in range(len(prompts))]
