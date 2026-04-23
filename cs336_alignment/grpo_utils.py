import torch
from typing import Callable, List, Dict, Tuple, Literal, Optional
import torch
from typing import List, Dict, Callable
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from transformers import PreTrainedModel
import numpy as np
import wandb
from vllm import LLM, SamplingParams


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    
    assert len(rollout_responses) == len(repeated_ground_truths), "Response 和 Ground Truth 数量必须一致"
    assert len(rollout_responses) % group_size == 0, "总样本数必须是 group_size 的整数倍"

    raw_rewards_list = []

    for response, truth in zip(rollout_responses, repeated_ground_truths):

        score_dict = reward_fn(response, truth)
        raw_rewards_list.append(score_dict["reward"])

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    num_questions = raw_rewards.shape[0] // group_size

    grouped_rewards = raw_rewards.view(num_questions, group_size) # shape: (num_questions, group_size)

    group_means = grouped_rewards.mean(dim=1, keepdim=True) # shape: (num_questions, 1)

    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        advantages = grouped_rewards - group_means

    advantages = advantages.view(-1)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
        "mean_advantage": advantages.mean().item(),
    }

    return advantages, raw_rewards, metadata