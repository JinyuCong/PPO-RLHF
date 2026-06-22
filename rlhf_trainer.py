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
from transformers import AutoTokenizer
from typing import Optional
import datetime
import os
from tqdm import tqdm
import itertools

from config import Config
from model import ActorModel, CriticModel, ReferenceModel, load_models
from reward_model import RewardModel
from data import get_prompt_dataloader, load_prompt_data
from ppo_trainer import collect_rollouts, compute_advantages, ppo_update
from utils import set_seed, save_checkpoint, load_checkpoint, get_logger, masked_mean, compute_kl_divergence


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
        set_seed(config.training.seed)
        # TODO: 加载 actor, critic, ref_model
        actor, critic, ref_model = load_models(config.model, device=self.device)
        self.actor: ActorModel = actor
        self.critic: CriticModel = critic
        self.ref_model: ReferenceModel = ref_model

        # TODO: 加载奖励模型（从 checkpoint 加载，路径在 config.reward.reward_model_save_path）
        self.reward_model: RewardModel = RewardModel(config.reward).to(self.device)
        rm_path = os.path.join(config.reward.reward_model_save_path, "reward_model.pt")
        state_dict = torch.load(rm_path, map_location=self.device)
        self.reward_model.load_state_dict(state_dict)
        self.reward_model.eval()

        # TODO: 初始化优化器
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.ppo.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config.ppo.critic_lr)

        # TODO: 加载 prompt 数据集
        prompts = load_prompt_data(config.training.data_path)
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        self.prompt_dataloader = get_prompt_dataloader(prompts=prompts, 
                                                       tokenizer=tokenizer, 
                                                       max_length=config.model.max_length, 
                                                       batch_size=8)

        # TODO: 初始化 TensorBoard writer
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer: Optional[SummaryWriter] = SummaryWriter(f"runs/{timestamp}")

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
        data_iter = itertools.cycle(self.prompt_dataloader)
        pbar = tqdm(total=self.config.ppo.total_steps, initial=self.global_step)
        
        while self.global_step < self.config.ppo.total_steps:
            batch = next(data_iter)
            metrics = self._rollout_and_update(batch)
            self.log_metrics(metrics)
            self.global_step += 1
            pbar.update(1)
            
            # 定期保存
            if self.global_step % self.config.training.save_steps == 0:
                self.save()
            
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
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # 生成回复，收集数据
        buffer = collect_rollouts(
            actor=self.actor,
            critic=self.critic,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            config=self.config.ppo,
            device=self.device
        )
        
        # GAE 计算优势函数
        buffer = compute_advantages(buffer=buffer, config=self.config.ppo)
        
        # 计算平均masked reward
        rewards = buffer.rewards
        prompt_len = buffer.prompt_ids.size(1)
        response_mask = buffer.attention_mask[:, prompt_len:].to(rewards.dtype)
        avg_reward = masked_mean(rewards, response_mask)
        
        old_log_probs = buffer.old_log_probs
        ref_log_probs = buffer.ref_log_probs
        
        # 多轮更新 actor 和 critic
        metrics = ppo_update(
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            buffer=buffer,
            config=self.config.ppo
        )
        
        kl = compute_kl_divergence(old_log_probs, ref_log_probs, response_mask)  # (B,)
        avg_kl = kl.mean()
        
        metrics["reward/mean"] = avg_reward.item()
        metrics["kl/mean"] = avg_kl.item()
        
        return metrics

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
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.global_step)
        
        # 每 log_steps 步打印一行
        if self.global_step % self.config.training.log_steps == 0:
            msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[step {self.global_step}] {msg}")

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
        save_checkpoint(
            save_dir=self.config.training.output_dir,
            step=self.global_step,
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
        )

    def load(self, checkpoint_path: str):
        """
        加载 checkpoint，用于断点续训

        TODO: 实现加载逻辑，注意恢复 global_step
        """
        # load_checkpoint 会就地恢复 critic / 两个 optimizer / actor 权重，并返回步数
        self.global_step = load_checkpoint(
            load_dir=checkpoint_path,
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
        )
