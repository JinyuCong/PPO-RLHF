# =============================================================================
# ppo_trainer.py —— PPO 算法核心实现
#
# PPO 的一次完整迭代分为两个阶段：
#
# ━━━ 阶段一：Rollout（数据收集）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   用当前 Actor 生成一批回复，记录：
#     - 生成的 token 序列
#     - 每个 token 的 log 概率（来自 Actor）
#     - 参考策略的 log 概率（来自 Reference Model，用于 KL 惩罚）
#     - 奖励模型给的分数（只在序列末尾有奖励）
#     - Critic 估计的状态价值
#
# ━━━ 阶段二：Update（参数更新）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   用收集的数据计算优势函数（GAE），然后多次迭代更新 Actor 和 Critic：
#     - Actor Loss：PPO-clip 目标函数
#     - Critic Loss：价值函数的 MSE 损失
#     - 熵奖励：鼓励探索
# =============================================================================

import torch
import torch.nn.functional as F
from typing import Dict, List
from dataclasses import dataclass

from model import ActorModel, CriticModel, ReferenceModel
from reward_model import RewardModel
from config import PPOConfig


@dataclass
class RolloutBuffer:
    """
    存储一次 rollout 收集到的所有数据

    字段说明：
      prompt_ids:      (batch, prompt_len) —— 输入 prompt 的 token ids
      response_ids:    (batch, resp_len)   —— 生成的 response token ids
      attention_mask:  (batch, total_len)  —— 完整序列的 mask（prompt + response）

      old_log_probs:   (batch, resp_len)   —— rollout 时 Actor 的 log π(a|s)
                                               PPO 更新时用作"旧策略"参考
      ref_log_probs:   (batch, resp_len)   —— Reference Model 的 log π_ref(a|s)
                                               用于计算 KL 惩罚

      rewards:         (batch, resp_len)   —— 每个 token 的即时奖励
                                               通常只在最后一个 token 有非零奖励
                                               格式：[0, 0, ..., 0, r_final - kl_penalty]

      values:          (batch, resp_len)   —— Critic 估计的 V(s_t)
      advantages:      (batch, resp_len)   —— GAE 计算出的优势函数 A_t
      returns:         (batch, resp_len)   —— 折扣回报 G_t（用于训练 Critic）
    """
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    attention_mask: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor = None  # 在 compute_advantages 后填充
    returns: torch.Tensor = None     # 在 compute_advantages 后填充


def collect_rollouts(actor: ActorModel, critic: CriticModel,
                     ref_model: ReferenceModel, reward_model: RewardModel,
                     prompt_ids: torch.Tensor, attention_mask: torch.Tensor,
                     config: PPOConfig, device: str) -> RolloutBuffer:
    """
    ━━━ 阶段一：Rollout 数据收集 ━━━

    完整流程：
      1. Actor 生成 response
      2. 构建完整序列（prompt + response）
      3. 计算 Actor 的 log 概率（old_log_probs）
      4. 计算 Reference 的 log 概率（ref_log_probs）
      5. 计算 KL 惩罚：kl = old_log_probs - ref_log_probs （逐 token）
      6. 奖励模型对完整序列打分（只是一个末尾分数）
      7. 构建 per-token 奖励：最后一个有效 token 放奖励分数，其他位置放 -kl_coef * kl
         这样 KL 惩罚均摊到每个生成的 token 上
      8. Critic 估计每个 token 位置的价值

    所有操作都在 @torch.no_grad() 下进行（rollout 不需要梯度）

    返回：填充了除 advantages/returns 之外所有字段的 RolloutBuffer

    TODO: 实现上述 8 步
    提示：
      - actor.generate() 返回的是 prompt+response 的完整序列
      - 需要从完整序列中分离出 response 部分（从 prompt_len 之后）
      - attention_mask 对应完整序列
    """
    with torch.no_grad():
        # Actor 生成 response，构建完整序列（prompt + response）
        prompt_response_ids = actor.generate(input_ids=prompt_ids, 
                                             attention_mask=attention_mask)
        prompt_len = prompt_ids.size(1)
        response_ids = prompt_response_ids[:, prompt_len:]
        
        # 构建 response attention mask
        eos_token_id = actor.tokenizer.eos_token_id
        is_eos = (response_ids == eos_token_id)  # [0, ..., 0, 1, 1, ...] 1的位置为是eos token的位置，需要将这个反过来 [1, ..., 1, 0, 0, ...]
        eos_cumsum = is_eos.long().cumsum(dim=-1)
        response_attention_mask = (eos_cumsum <= 1).to(attention_mask.dtype)
        
        full_attention_mask = torch.concat([attention_mask, response_attention_mask], dim=-1)  # (B, prompt_len + response_len)
        
        # 计算 Actor 的 log 概率（old_log_probs）
        old_log_probs = actor.get_log_probs(input_ids=prompt_response_ids,  # (B, response_len)
                                            attention_mask=full_attention_mask)[:, prompt_len-1:]
        # 计算 Reference 的 log 概率（ref_log_probs）
        ref_log_probs = ref_model.get_log_probs(input_ids=prompt_response_ids,  # (B, response_len)
                                                attention_mask=full_attention_mask)[:, prompt_len-1:]
        # 计算 KL 惩罚：kl = old_log_probs - ref_log_probs （逐 token）
        kl = old_log_probs - ref_log_probs
        
        # 用奖励模型计算这个prompt和response的奖励
        reward = reward_model(input_ids=prompt_response_ids,  # (B,)
                              attention_mask=full_attention_mask)
        # 中间所有token的奖励（都为负）
        per_token_reward = -config.kl_coef * kl  # (B, response_len)
        
        # 找到每行 response 最后一个有效 token 的位置（不一定是 -1，因为可能提前 EOS 后面是 padding）
        last_valid_idx = response_attention_mask.sum(dim=-1) - 1  # (B,)
        B = per_token_reward.size(0)
        per_token_reward[torch.arange(B, device=per_token_reward.device), last_valid_idx] += reward
        
        # Critic 估计每个 token 位置的价值
        values = critic(input_ids=prompt_response_ids,  # (B, response_len)
                        attention_mask=full_attention_mask)[:, prompt_len-1:-1]
        
        buffer = RolloutBuffer(
            prompt_ids=prompt_ids.to(device),
            response_ids=response_ids.to(device),
            attention_mask=full_attention_mask.to(device),
            old_log_probs=old_log_probs.to(device),
            ref_log_probs=ref_log_probs.to(device),
            rewards=per_token_reward.to(device),
            values=values.to(device),
        )
        
    
    return buffer


def compute_advantages(buffer: RolloutBuffer, config: PPOConfig) -> RolloutBuffer:
    """
    ━━━ GAE（广义优势估计）━━━

    GAE 公式：
      δ_t = r_t + γ * V(s_{t+1}) - V(s_t)          ← TD 误差
      A_t = δ_t + γ * λ * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...

    等价的递推公式（从后往前计算，更高效）：
      A_T = δ_T                                      ← 最后一步
      A_t = δ_t + γ * λ * A_{t+1}                   ← 往前递推

    折扣回报（用于训练 Critic）：
      G_t = A_t + V(s_t)

    参数：
      buffer: 已填充 rewards 和 values 的 RolloutBuffer

    步骤：
      1. 从 buffer 取出 rewards: (batch, T) 和 values: (batch, T)
      2. 计算每步 TD 误差：delta_t = r_t + gamma * V_{t+1} - V_t
         注意：最后一步 V_{T+1} = 0（序列结束后没有价值）
      3. 从后往前递推计算 advantages
      4. 归一化 advantages（减均值除标准差），有助于训练稳定
         注意：只对有效 token 位置做归一化（不包括 prompt 部分）
      5. returns = advantages + values

    返回：填充了 advantages 和 returns 的 RolloutBuffer

    TODO: 实现上述步骤
    提示：可以用 torch.flip 把序列反转，方便从后往前循环
    """
    raise NotImplementedError


def compute_ppo_loss(actor: ActorModel, critic: CriticModel,
                     buffer: RolloutBuffer, config: PPOConfig):
    """
    ━━━ 阶段二：PPO 更新的损失计算 ━━━

    包含三部分损失：

    1. Actor Loss（PPO-clip）：
       r_t = exp(log_probs_new - log_probs_old)     ← 重要性采样比率
       L_clip = -mean(min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t))

       直觉：
         - 如果 A_t > 0（这个动作好），希望 r_t 大（增大新策略概率）
           但 clip 限制最多增大到 1+ε
         - 如果 A_t < 0（这个动作不好），希望 r_t 小（减小新策略概率）
           但 clip 限制最多减小到 1-ε

    2. Critic Loss（价值函数回归）：
       L_value = mean((V_new - returns)^2)
       可选：也可以对 Critic 的更新做 clip（和 PPO 论文一致）

    3. 熵奖励（鼓励探索）：
       entropy = -sum(π * log π)（对词表上的分布）
       L_entropy = -mean(entropy)   ← 负号是因为我们想最大化熵

    总损失：
       loss = L_clip + vf_coef * L_value + ent_coef * L_entropy

    参数：
      buffer: 完整的 RolloutBuffer（包含 advantages 和 returns）

    返回：
      total_loss, actor_loss, critic_loss, entropy

    TODO: 实现上述计算
    提示：
      - 用 buffer 中的完整序列（prompt + response）重新过 actor 和 critic
      - log_probs_new 只取 response 部分（和 old_log_probs 对齐）
      - 注意 mask：只对有效的 response token 计算损失，padding 位置不计算
    """
    raise NotImplementedError


def ppo_update(actor: ActorModel, critic: CriticModel,
               actor_optimizer, critic_optimizer,
               buffer: RolloutBuffer, config: PPOConfig):
    """
    执行一次 PPO 更新：在 buffer 数据上迭代 ppo_epochs 次

    步骤：
      1. 外层循环：重复 config.ppo_epochs 次
      2. （可选）把 buffer 数据随机打乱，分成多个 mini-batch
      3. 对每个 mini-batch：
         a. 调用 compute_ppo_loss 计算损失
         b. 梯度清零、反向传播、梯度裁剪、参数更新
         c. 记录损失数值（用于日志）

    返回：
      各种损失的平均值（字典格式，方便日志记录）

    TODO: 实现上述步骤
    提示：
      - 梯度裁剪：nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
      - 可以用 torch.utils.data.TensorDataset + DataLoader 来做 mini-batch 切分
    """
    raise NotImplementedError


if __name__ == "__main__":
    from config import Config
    from transformers import AutoTokenizer, PreTrainedTokenizer
    cfg = Config()
    
    actor = ActorModel(cfg.model).to('cuda')
    critic = CriticModel(cfg.model).to('cuda')
    ref_model = ReferenceModel(cfg.model).to('cuda')
    reward_model = RewardModel(cfg.reward).to('cuda')
    
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    test_prompt = "I am a man. Who are you ?"
    tokenized = tokenizer(test_prompt, padding='max_length', 
                          max_length=cfg.model.max_length, truncation=True, 
                          return_tensors='pt')
    input_ids = tokenized['input_ids'].to('cuda')
    attention_mask = tokenized['attention_mask'].to('cuda')
    
    collect_rollouts(actor, critic, 
                     ref_model, reward_model, 
                     input_ids, attention_mask,
                     cfg, 'cuda')
    
    