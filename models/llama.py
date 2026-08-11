"""
Everything specific to Llama lives here.

This file handles:
- Llama model loading
- Llama tokenizer
- Llama chat template
- sequence generation
- batch generation
"""

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


def load(
    model_path,
    torch_dtype="bfloat16",
    device_map="auto"
):

    print(
        f"[llama] loading tokenizer from: {model_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    print(
        f"[llama] loading model weights from: {model_path}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, torch_dtype),
        device_map=device_map
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _build_messages(prompt):

    return [
        {
            "role": "user",
            "content": prompt
        }
    ]


def generate_one(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100
):

    messages = _build_messages(prompt)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(
        output[0][prompt_len:],
        skip_special_tokens=True
    ).strip()


def generate_batch(
    model,
    tokenizer,
    prompts,
    max_new_tokens=100
):

    tokenizer.padding_side = "left"

    messages_list = [
        _build_messages(prompt)
        for prompt in prompts
    ]

    formatted_prompts = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        for messages in messages_list
    ]

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    results = []

    for i in range(len(prompts)):

        results.append(
            tokenizer.decode(
                output[i][prompt_len:],
                skip_special_tokens=True
            ).strip()
        )

    return results
