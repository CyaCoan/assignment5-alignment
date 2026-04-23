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

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    
    weighted_log_probs = raw_rewards_or_advantages * policy_log_probs

    loss = -weighted_log_probs

    return loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages

    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    surr2 = clipped_ratio * advantages

    loss = -torch.min(surr1, surr2)

    with torch.no_grad():

        clipped_mask = (surr2 < surr1).float()
        clip_fraction = clipped_mask.mean()
        
        metadata = {
            "clip_fraction": clip_fraction, # 非常关键：若此值接近 1.0，说明学习率过高或模型已停止学习
            "ratio_mean": ratio.mean(),
            "ratio_max": ratio.max(),
            "ratio_min": ratio.min(),
        }

    return loss, metadata

def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    loss = -(ratio * advantages)
    
    with torch.no_grad():

        surr1 = ratio * advantages

        cliprange = 0.2
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        surr2 = clipped_ratio * advantages

        clipped_mask = (surr2 < surr1).float()
        clip_fraction = clipped_mask.mean()
        
        metadata = {
            "ratio_mean": ratio.mean(),
            "ratio_max": ratio.max(),
            "ratio_min": ratio.min(),
            "clip_fraction": clip_fraction,
        }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    metadata = {}

    if loss_type == "no_baseline":

        assert raw_rewards is not None, "no_baseline 模式必须提供 raw_rewards"
        
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        )

    elif loss_type == "reinforce_with_baseline":
        
        assert advantages is not None, "reinforce_with_baseline 模式必须提供 advantages"
        
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        )

    elif loss_type == "grpo_clip":
        
        assert advantages is not None, "grpo_clip 模式必须提供 advantages"
        assert old_log_probs is not None, "grpo_clip 模式必须提供 old_log_probs"
        assert cliprange is not None, "grpo_clip 模式必须提供 cliprange"
        
        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )

        metadata.update(grpo_metadata)

    elif loss_type == "grpo_no_clip":

        assert advantages is not None and old_log_probs is not None

        loss, grpo_metadata = compute_grpo_no_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs
        )
        
        metadata.update(grpo_metadata)

    else:

        raise ValueError(f"不支持的 loss_type: {loss_type}")

    return loss, metadata