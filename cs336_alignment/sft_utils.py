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

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    
    masked_tensor = tensor * mask

    if dim is None:
        total_sum = torch.sum(masked_tensor)
    else:
        total_sum = torch.sum(masked_tensor, dim=dim)

    return total_sum / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    batch_size = policy_log_probs.shape[0]

    nll_per_token = -policy_log_probs

    total_masked_loss = masked_normalize(
        tensor=nll_per_token, 
        mask=response_mask, 
        normalize_constant=normalize_constant, 
        dim=None
    )

    microbatch_loss_mean = total_masked_loss / batch_size
    scaled_loss = microbatch_loss_mean / gradient_accumulation_steps

    scaled_loss.backward()

    metadata = {"loss": microbatch_loss_mean}

    return scaled_loss, metadata

def log_generations(
    vllm_model: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    step: int,
    log_prefix: str = "eval"
):
    """
    让模型生成回答并记录详细的评估指标。
    """
    # 1. 模型生成回答
    # 注意：在调用此函数前，应确保已将最新的 policy 权重加载到了 vLLM 实例中
    outputs = vllm_model.generate(prompts, sampling_params)
    
    table_data = []
    
    # 用于统计的数据
    all_lengths = []
    correct_lengths = []
    incorrect_lengths = []
    total_reward = 0
    total_format_reward = 0
    total_answer_reward = 0
    
    # 2. 逐条处理生成结果
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answer = ground_truths[i]
        
        # 计算奖励
        scores = reward_fn(generated_text, gold_answer)
        
        r = scores.get("reward", 0.0)
        fr = scores.get("format_reward", 0.0)
        ar = scores.get("answer_reward", 0.0)
        
        # 计算响应长度
        resp_len = len(generated_text)
        all_lengths.append(resp_len)
        
        if r > 0.5: # 认为是正确的
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)
            
        total_reward += r
        total_format_reward += fr
        total_answer_reward += ar

        # 准备存入 wandb Table 的数据（展示前几条即可，防止日志过大）
        if i < 100: 
            table_data.append([
                step, 
                prompts[i], # 只取 prompt 结尾部分
                generated_text, 
                gold_answer, 
                r, fr, ar
            ])

    # 3. 计算聚合统计量
    metrics = {
        f"{log_prefix}/accuracy": total_reward / len(prompts),
        f"{log_prefix}/format_score": total_format_reward / len(prompts),
        f"{log_prefix}/answer_score": total_answer_reward / len(prompts),
        f"{log_prefix}/avg_length": np.mean(all_lengths),
        f"{log_prefix}/avg_length_correct": np.mean(correct_lengths) if correct_lengths else 0,
        f"{log_prefix}/avg_length_incorrect": np.mean(incorrect_lengths) if incorrect_lengths else 0,
    }

    # 4. 记录到日志系统
    if wandb.run is not None:
        # 记录表格：方便直接在网页看具体的推理逻辑
        columns = ["step", "prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"]
        wandb.log({f"{log_prefix}/samples": wandb.Table(columns=columns, data=table_data)}, step=step)
        # 记录标量数值
        wandb.log(metrics, step=step)
    
    print(f"Step {step}: Accuracy: {metrics[f'{log_prefix}/accuracy']:.4f}, Avg Len: {metrics[f'{log_prefix}/avg_length']:.1f}")

    return metrics