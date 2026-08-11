"""
Everything specific to Gemma lives here.

This file handles:

- Gemma model loading
- Gemma tokenizer loading
- Gemma chat template
- Gemma sequence generation
- Gemma batch generation
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(
            torch,
            torch_dtype
        ),
        device_map=device_map
    )

    model.eval()

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _build_messages(prompt):

    """
    Gemma receives the prompt as a user message.

    The pipeline only supplies a plain string.
    Gemma handles its own chat-message format here.
    """

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

    messages = _build_messages(
        prompt
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    prompt_len = inputs[
        "input_ids"
    ].shape[-1]

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_tokens = output[
        0
    ][prompt_len:]

    return tokenizer.decode(
        generated_tokens,
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

    prompt_len = inputs[
        "input_ids"
    ].shape[1]

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    results = []

    for i in range(
        len(prompts)
    ):

        generated_tokens = output[
            i
        ][prompt_len:]

        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        results.append(
            generated_text
        )

    return results
