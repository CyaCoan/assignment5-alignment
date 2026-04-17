import torch
import wandb
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Callable
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    
    all_input_ids = []
    all_labels = []
    all_response_masks = []
    all_lengths = []

    for p_str, o_str in zip(prompt_strs, output_strs):

        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)
        combined_ids = p_ids + o_ids

        all_input_ids.append(combined_ids)
        all_lengths.append(len(combined_ids))

        mask = [0] * len(p_ids) + [1] * len(o_ids)
        all_response_masks.append(mask)

    max_len = max(all_lengths)
    batch_size = len(prompt_strs)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (ids, m) in enumerate(zip(all_input_ids, all_response_masks)):

        length = len(ids)
        padded_input_ids[i, :length] = torch.tensor(ids)
        padded_masks[i, :length] = torch.tensor(m)

    input_ids = padded_input_ids[:, :-1]
    labels = padded_input_ids[:, 1:].clone()
    response_mask = padded_masks[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:

    lse = torch.logsumexp(logits, dim=-1)

    probs = F.softmax(logits, dim=-1)

    exp_logits = torch.sum(probs * logits, dim=-1)

    entropy = lse - exp_logits

    return entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    
    outputs = model(input_ids)
    logits = outputs.logits

    all_log_probs = F.log_softmax(logits, dim=-1)

    log_probs = torch.gather(all_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result