"""
Gemma-specific implementation.

This file contains:
- Gemma model loading
- Gemma tokenizer
- Gemma chat formatting
- batch generation
"""

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


# ========================================================
# PRIVATE GEMMA FORMATTER
# ========================================================

class _GemmaFormatter:

    def build_messages(self, prompt):

        return [
            {
                "role": "user",
                "content": prompt
            }
        ]


# Create the private formatter
_formatter = _GemmaFormatter()


# ========================================================
# LOAD
# ========================================================

def load(
    model_path,
    torch_dtype="bfloat16",
    device_map="auto"
):

    print(
        f"[gemma] loading tokenizer from: "
        f"{model_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    print(
        f"[gemma] loading model weights from: "
        f"{model_path}"
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

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    return model, tokenizer


# ========================================================
# BATCH GENERATION
# ========================================================

def generate_batch(
    model,
    tokenizer,
    prompts,
    max_new_tokens=100
):

    tokenizer.padding_side = "left"

    # ----------------------------------------------------
    # Build Gemma messages
    # ----------------------------------------------------

    messages_list = [
        _formatter.build_messages(prompt)
        for prompt in prompts
    ]

    # ----------------------------------------------------
    # Apply Gemma chat template
    # ----------------------------------------------------

    formatted_prompts = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        for messages in messages_list
    ]

    # ----------------------------------------------------
    # Tokenize
    # ----------------------------------------------------

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    prompt_len = inputs[
        "input_ids"
    ].shape[1]

    # ----------------------------------------------------
    # Generate
    # ----------------------------------------------------

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # ----------------------------------------------------
    # Decode
    # ----------------------------------------------------

    results = []

    for i in range(
        len(prompts)
    ):

        generated_tokens = output[
            i
        ][prompt_len:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        results.append(
            response
        )

    return results
