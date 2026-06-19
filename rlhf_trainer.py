# =============================================================================
# rlhf_trainer.py —— RLHF 整体训练流程的协调器
#
# 这个文件把所有模块粘合在一起，负责：
#   1. 初始化所有模型、优化器、数据加载器
#   2. 主训练循环：rollout → compute_advantages → ppo_update → log → checkpoint
#   3. 监控训练状态（KL 散度、奖励均值、损失曲线）
#
# 你应该在这里实现 "训练的节奏"，具体算法细节在 ppo_trainer.py
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import os

from config import Config
from model import ActorModel, CriticModel, ReferenceModel, load_models
from reward_model import RewardModel
from data import get_prompt_dataloader, load_prompt_data
from ppo_trainer import collect_rollouts, compute_advantages, ppo_update
from utils import set_seed, save_checkpoint, load_checkpoint, get_logger


class RLHFTrainer:
    """
    RLHF 训练器：管理整个 PPO RLHF 训练过程

    使用方式：
      trainer = RLHFTrainer(config)
      trainer.train()
    """

    def __init__(self, config: Config):
        """
        初始化训练器

        TODO 步骤：
          1. 保存 config
          2. 设置随机种子（utils.set_seed）
          3. 设置 device
          4. 加载模型（load_models）
          5. 加载奖励模型（RewardModel）并设为 eval 模式（不参与训练）
          6. 初始化 Actor 和 Critic 的优化器（AdamW）
          7. 加载 prompt 数据集和 DataLoader
          8. 初始化 TensorBoard SummaryWriter
          9. 初始化 logger

        注意：
          - 奖励模型参数也要冻结（eval + no_grad）
          - Reference Model 参数要冻结
          - 只有 Actor 和 Critic 有 optimizer
        """
        self.config = config
        self.device = config.training.device

        # TODO: 设置随机种子
        # TODO: 加载 actor, critic, ref_model
        self.actor: ActorModel = None
        self.critic: CriticModel = None
        self.ref_model: ReferenceModel = None

        # TODO: 加载奖励模型（从 checkpoint 加载，路径在 config.reward.reward_model_save_path）
        self.reward_model: RewardModel = None

        # TODO: 初始化优化器
        self.actor_optimizer = None
        self.critic_optimizer = None

        # TODO: 加载 prompt 数据集
        self.prompt_dataloader = None

        # TODO: 初始化 TensorBoard writer
        self.writer: Optional[SummaryWriter] = None

        # 训练步数计数器
        self.global_step = 0

    def train(self):
        """
        主训练循环

        外层循环：遍历 prompt dataloader，每个 batch 是一组 prompts
        对每个 batch：
          1. collect_rollouts()         ← 生成回复，收集数据
          2. compute_advantages()       ← GAE 计算优势函数
          3. ppo_update()               ← 多轮更新 actor 和 critic
          4. log_metrics()              ← 记录训练指标
          5. 检查是否需要保存 checkpoint

        循环直到 global_step >= config.ppo.total_steps

        TODO: 实现主循环
        提示：
          - prompt dataloader 可能不够长，需要用 itertools.cycle 无限循环
          - 每次迭代后 self.global_step += 1
          - 用 tqdm 包装循环可以显示进度条
        """
        raise NotImplementedError

    def _rollout_and_update(self, batch):
        """
        一次完整的 rollout + 更新流程（主循环的核心）

        步骤：
          1. 从 batch 取出 input_ids, attention_mask，移到 device
          2. 调用 collect_rollouts(...) 得到 buffer
          3. 调用 compute_advantages(buffer, config.ppo) 填充 advantages/returns
          4. 调用 ppo_update(...) 执行参数更新
          5. 返回更新后的指标（loss, reward, kl 等）

        TODO: 实现上述步骤
        """
        raise NotImplementedError

    def log_metrics(self, metrics: dict):
        """
        记录训练指标到 TensorBoard 和控制台

        metrics 应该包含：
          - "reward/mean":    这批 rollout 的平均奖励分数
          - "kl/mean":        平均 KL 散度（监控偏离 ref 的程度）
          - "loss/actor":     Actor 的 PPO-clip 损失
          - "loss/critic":    Critic 的价值回归损失
          - "loss/entropy":   策略熵（越大代表越"随机"、探索性越强）
          - "loss/total":     总损失

        TODO:
          1. 用 self.writer.add_scalar(key, value, self.global_step) 写入 TensorBoard
          2. 每 log_steps 步打印一行到控制台
        """
        raise NotImplementedError

    def save(self):
        """
        保存 checkpoint

        需要保存：
          - actor 的模型权重和 tokenizer
          - critic 的模型权重
          - actor_optimizer 状态
          - critic_optimizer 状态
          - global_step（用于断点续训）

        TODO: 调用 utils.save_checkpoint 或直接实现保存逻辑
        提示：torch.save(state_dict, path)
              actor.model.save_pretrained(path) 可以保存为 HuggingFace 格式
        """
        raise NotImplementedError

    def load(self, checkpoint_path: str):
        """
        加载 checkpoint，用于断点续训

        TODO: 实现加载逻辑，注意恢复 global_step
        """
        raise NotImplementedError
