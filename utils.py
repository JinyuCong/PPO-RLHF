# =============================================================================
# utils.py —— 工具函数集合
#
# 这里放那些不属于任何特定模块、但到处都会用到的辅助函数
# =============================================================================

import os
import random
import logging
import numpy as np
import torch
from typing import Dict, Any


def set_seed(seed: int):
	"""
	固定所有随机种子，保证实验可复现

	需要固定的地方：
	  - Python random 模块
	  - NumPy
	  - PyTorch CPU
	  - PyTorch CUDA（所有 GPU）
	  - 还可以设置 torch.backends.cudnn.deterministic = True
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)   # 覆盖所有 GPU，单卡也适用
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False  # deterministic=True 时必须关掉


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
	"""
	创建并配置一个 logger

	行为：
	  - 同时输出到控制台（StreamHandler）
	  - 如果提供了 log_dir，也写入文件（FileHandler）
	  - 格式：[时间] [级别] 消息
	"""
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)

	# Handler：输出到控制台
	handler = logging.StreamHandler()
	handler.setLevel(logging.DEBUG)

	formatter = logging.Formatter(
		"[%(asctime)s] [%(levelname)s] %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	if log_dir:
		from logging.handlers import RotatingFileHandler
		os.makedirs(log_dir, exist_ok=True)
		file_handler = RotatingFileHandler(
			os.path.join(log_dir, f"{name}.log"),
			maxBytes=5*1024*1024, backupCount=3
		)
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	return logger


def save_checkpoint(save_dir: str, step: int, actor, critic,
	actor_optimizer, critic_optimizer):
	"""
	保存训练 checkpoint

	文件结构：
	  save_dir/
	    step_{step}/
	      actor/              ← HuggingFace 格式
	      critic.pt           ← Critic state_dict
	      optimizers.pt       ← 两个优化器 state_dict
	      training_state.pt   ← global_step
	"""
	ckpt_dir = os.path.join(save_dir, f"step_{step}")

	# Actor：用 HuggingFace 格式保存，方便后续直接 from_pretrained 加载推理
	actor_dir = os.path.join(ckpt_dir, "actor")
	os.makedirs(actor_dir, exist_ok=True)
	actor.model.save_pretrained(actor_dir)
	actor.tokenizer.save_pretrained(actor_dir)

	# Critic：只保存 state_dict（不是 HuggingFace 模型，用 torch.save 即可）
	torch.save(
		critic.state_dict(),
		os.path.join(ckpt_dir, "critic.pt")
	)

	# 优化器：保存两个 optimizer 的状态（包含 momentum、学习率等），用于断点续训
	torch.save(
		{
			"actor_optimizer": actor_optimizer.state_dict(),
			"critic_optimizer": critic_optimizer.state_dict(),
		},
		os.path.join(ckpt_dir, "optimizers.pt")
	)

	# 训练状态：保存当前步数，续训时恢复 global_step
	torch.save(
		{"step": step},
		os.path.join(ckpt_dir, "training_state.pt")
	)


def load_checkpoint(load_dir: str, actor, critic, actor_optimizer, critic_optimizer):
	"""
	加载 checkpoint，用于断点续训

	返回：加载成功后的 global_step
	"""
	# Critic
	critic.load_state_dict(
		torch.load(
			os.path.join(load_dir, "critic.pt"),
			map_location=next(critic.parameters()).device  # 自动匹配当前设备
		)
	)

	# 优化器
	opt_state = torch.load(
		os.path.join(load_dir, "optimizers.pt"),
		map_location="cpu"  # optimizer state 先加载到 CPU，框架会自动处理
	)
	actor_optimizer.load_state_dict(opt_state["actor_optimizer"])
	critic_optimizer.load_state_dict(opt_state["critic_optimizer"])

	# Actor：用 from_pretrained 加载，需要在外部把权重赋给 actor.model
	# 这里只返回路径，让调用方自行决定是否重新加载整个 actor（通常在 RLHFTrainer.load 里做）
	actor_dir = os.path.join(load_dir, "actor")
	actor.model = actor.model.from_pretrained(actor_dir).to(next(actor.model.parameters()).device)

	# 恢复训练步数
	state = torch.load(os.path.join(load_dir, "training_state.pt"), map_location="cpu")
	return state["step"]


def compute_kl_divergence(log_probs_new: torch.Tensor,
	log_probs_ref: torch.Tensor,
	mask: torch.Tensor = None) -> torch.Tensor:
	"""
	计算新策略和参考策略之间的 KL 散度

	近似公式（逐 token）：
	  KL(π_new || π_ref) ≈ log π_new(a) - log π_ref(a)

	这是一个 token-level 的 KL 近似（不是完整分布上的 KL），
	在 RLHF 中被广泛使用，计算简单且效果好。

	参数：
	  log_probs_new: (batch, seq_len) —— 新策略的 log 概率
	  log_probs_ref: (batch, seq_len) —— 参考策略的 log 概率
	  mask:          (batch, seq_len) —— 有效 token 的 mask（可选）

	返回：
	  kl: (batch,) —— 每条序列的平均 KL 散度
	"""
	token_kl = log_probs_new - log_probs_ref
	if mask is not None:
		return masked_mean(token_kl, mask, dim=-1)
	return token_kl.mean(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
	"""
	带 mask 的均值计算（忽略 padding 位置）

	这是一个常用的工具函数，在计算 loss、reward、KL 时都会用到

	参数：
	  tensor: 需要求均值的张量
	  mask:   同形状的 0/1 mask，1 表示有效位置
	  dim:    在哪个维度求均值

	公式：
	  masked_mean = sum(tensor * mask, dim) / sum(mask, dim)
	"""
	return torch.sum(tensor * mask, dim=dim) / (torch.sum(mask, dim=dim) + 1e-8)


def whiten(tensor: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
	"""
	白化（标准化）张量：减均值、除标准差

	在 RLHF 中用于归一化 advantages，让训练更稳定
	"""
	if mask is not None:
		mean = masked_mean(tensor, mask)
		# 对有效位置计算方差，需要手动展开
		variance = masked_mean((tensor - mean.unsqueeze(-1)) ** 2, mask)
		std = variance.sqrt()
		return (tensor - mean.unsqueeze(-1)) / (std.unsqueeze(-1) + 1e-8)
	return (tensor - tensor.mean(-1, keepdim=True)) / (tensor.std(-1, keepdim=True) + 1e-8)


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
	"""
	打印模型参数量信息（方便调试和日志）

	输出格式示例：
	  Model: ActorModel
	    Total parameters:     117,000,000
	    Trainable parameters: 117,000,000
	"""
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model: {model_name}")
	print(f"  Total parameters:     {total:,}")
	print(f"  Trainable parameters: {trainable:,}")
