# SPDX-License-Identifier: Apache-2.0
"""Shared, explicit prompting and stopping conventions for accuracy evaluation."""

import re

import torch

NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
RULER_GENERATION = {
    **{
        name: 128
        for name in (
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multivalue",
            "niah_multiquery",
        )
    },
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}


def _chat(tokenizer, prompt, model_type, dataset_format=False, *, ruler=False):
    if dataset_format and model_type == "llama" and "<|begin_of_text|>" in tokenizer.get_vocab():
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    options = {"enable_thinking": False} if model_type == "qwen3" else {}
    messages = [{"role": "user", "content": prompt}]
    if dataset_format and ruler and model_type == "qwen3":
        messages.insert(
            0,
            {"role": "system", "content": "Answer directly and concisely. Do not explain your reasoning."},
        )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)


def prepare_input(tokenizer, prompt, suite, dataset, model_type, style, cap, *, answer_prefix=""):
    """Format task inputs and return token IDs with truncation metadata."""
    if not isinstance(answer_prefix, str):
        raise ValueError("answer_prefix must be a string")
    suffix = answer_prefix if suite == "ruler" and not prompt.endswith(answer_prefix) else ""
    if style == "dataset":
        if suite == "longbench":
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            truncated = ids.shape[-1] > cap
            if truncated:
                half = cap // 2
                prompt = tokenizer.decode(ids[0, :half], skip_special_tokens=True) + tokenizer.decode(
                    ids[0, -half:], skip_special_tokens=True
                )
            if dataset not in NO_CHAT_TASKS and tokenizer.chat_template:
                prompt = _chat(tokenizer, prompt, model_type, dataset_format=True)
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            # The cap applies before wrapping; the final token count may be larger.
            return ids, truncated
        if not prompt.startswith(("<|begin", "<|im_start")) and tokenizer.chat_template:
            prompt = _chat(tokenizer, prompt, model_type, dataset_format=True, ruler=True)
        if model_type == "qwen3" and prompt.endswith("<|im_start|>assistant\n"):
            prompt += "<think>\n\n</think>\n\n"
        ids = tokenizer(prompt + suffix, return_tensors="pt").input_ids
    else:
        if style == "chat":
            prompt = _chat(tokenizer, prompt, model_type)
        ids = tokenizer(prompt + suffix, return_tensors="pt", add_special_tokens=style != "chat").input_ids
    truncated = ids.shape[-1] > cap
    if truncated:
        first = cap // 2
        ids = torch.cat((ids[:, :first], ids[:, -(cap - first) :]), dim=-1)
    return ids, truncated


def generation_options(tokenizer, suite, dataset, max_new, context_length):
    options = {
        "max_new_tokens": max_new,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    if suite == "longbench" and dataset == "samsum":
        newline = tokenizer.encode("\n", add_special_tokens=False)
        if not newline:
            raise ValueError("samsum requires a newline token")
        options.update(min_length=context_length + 1, eos_token_id=[tokenizer.eos_token_id, newline[-1]])
    return options


def postprocess(prediction, model_type):
    if model_type == "qwen3":
        prediction = re.sub(r"<think>.*?</think>\s*", "", prediction, flags=re.DOTALL)
        prediction = prediction.replace("<think>", "").strip()
    return prediction
