import torch
import wandb
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Callable
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
