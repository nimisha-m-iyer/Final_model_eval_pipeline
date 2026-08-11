"""
Everything specific to Gemma lives here, and only here.

Gemma's chat template does NOT accept a "system" role message -- its
template was only trained to understand "user" and "model" turns. So
any system prompt is merged into the start of the first user message
before formatting. This is the one Gemma-specific quirk; nothing else
differs from a standard Hugging Face chat model.
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


def _merge_system_into_user(messages):
    system = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if system and rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": "\n\n".join(system + [rest[0]["content"]])}
    return rest


def generate_one(model, tokenizer, messages, max_new_tokens=10):
    messages = _merge_system_into_user(messages)
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
    messages_list = [_merge_system_into_user(m) for m in messages_list]
    tokenizer.padding_side = "left"  # required for correct batched causal-LM generation

    prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages_list]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(output[i][prompt_len:], skip_special_tokens=True).strip() for i in range(len(prompts))]
