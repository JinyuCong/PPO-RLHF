# PPO-RLHF

从零实现的 **RLHF（基于人类反馈的强化学习）** 训练框架，用 PPO 算法对语言模型做对齐。
代码以 GPT2 为例，结构清晰、模块解耦，适合学习 RLHF / PPO 的完整流程。

---

## 🧩 RLHF 三阶段

```
阶段 1: SFT（监督微调）          —— 本项目外完成，直接用预训练 GPT2 作为起点
阶段 2: 训练奖励模型 Reward Model —— train_reward.py（也可直接用 HF 上预训练好的 RM）
阶段 3: PPO 强化学习             —— train_ppo.py
```

PPO 阶段同时存在 **4 个模型**：

| 模型 | 是否训练 | 作用 |
|------|---------|------|
| **Actor** | ✅ | 被优化的策略网络，负责生成回复 |
| **Critic** | ✅ | 价值网络，估计状态价值 V(s)，用于 GAE |
| **Reference** | ❌ 冻结 | SFT 模型副本，计算 KL 惩罚防止 Actor 偏离太远 |
| **Reward Model** | ❌ 冻结 | 给生成的回复打分 |

总奖励 = `reward_model_score − kl_coef × KL(π_actor ‖ π_ref)`

---

## 📁 项目结构

```
PPO_RLHF/
├── config.py          # 所有超参数集中管理（dataclass）
├── model.py           # Actor / Critic / Reference 三个模型定义
├── reward_model.py    # 奖励模型（自训练版 + HF 预训练版）、偏好损失
├── data.py            # 偏好数据集、Prompt 数据集、DataLoader
├── ppo_trainer.py     # PPO 核心算法：rollout、GAE、PPO-clip loss、update
├── rlhf_trainer.py    # 训练流程协调器：把所有模块串成主循环
├── utils.py           # 工具函数：seed、logger、checkpoint、KL、masked_mean
├── train_reward.py    # 入口①：训练奖励模型
└── train_ppo.py       # 入口②：PPO RLHF 主训练
```

---

## ⚙️ 环境依赖

```bash
pip install torch transformers datasets tensorboard tqdm numpy
```

- Python 3.10+
- 一块支持 CUDA 的 GPU（显存建议 ≥ 8GB）

---

## 🚀 快速开始

### 方式 A：使用 HuggingFace 预训练奖励模型（推荐，跳过阶段 2）

`config.py` 默认已配置好基于 GPT2 词表的预训练 RM：

```python
reward_model_name = "Ray2333/gpt2-large-harmless-reward_model"
```

直接跑 PPO：

```bash
python train_ppo.py
```

> 该 RM 用 GPT2 tokenizer，和 Actor 共享词表，无需任何 tokenizer 转换。

### 方式 B：自己训练奖励模型

```bash
# 1. 先训练奖励模型（数据集 Anthropic/hh-rlhf 会自动下载）
python train_reward.py
# → 产出 checkpoints/reward_model/reward_model.pt

# 2. 再跑 PPO（需把 rlhf_trainer.py 改回加载自训练 RM）
python train_ppo.py
```

### 监控训练

```bash
tensorboard --logdir ./logs
```

关注指标：
- `reward/mean` —— 平均奖励，应随训练**上升**
- `kl/mean` —— Actor 偏离 SFT 的程度，**不应爆炸式增长**
- `loss/actor` `loss/critic` `loss/entropy` —— 各项损失

---

## 🔧 关键配置（config.py）

| 参数 | 默认 | 说明 |
|------|------|------|
| `ppo.actor_lr` | 1e-6 | Actor 学习率（RL 阶段要很小）|
| `ppo.critic_lr` | 1e-5 | Critic 学习率，可比 actor 稍大 |
| `ppo.clip_eps` | 0.2 | PPO-clip 范围，越小更新越保守 |
| `ppo.gamma` | 0.99 | 折扣因子 |
| `ppo.gae_lambda` | 0.95 | GAE 的 λ，权衡 bias/variance |
| `ppo.kl_coef` | 0.1 | KL 惩罚系数，越大越贴近 SFT |
| `ppo.ent_coef` | 0.01 | 熵正则，鼓励探索 |
| `ppo.ppo_epochs` | 4 | 每批 rollout 数据复用几遍 |
| `ppo.total_steps` | 1000 | 总训练步数 |
| `ppo.max_new_tokens` | 128 | 生成回复的最大长度 |

---

## 🧠 算法要点

- **Rollout**：Actor 生成回复 → 记录 log 概率、参考策略 log 概率、奖励、价值估计
- **GAE**：`A_t = δ_t + γλ·A_{t+1}` 反向递推优势，`G_t = A_t + V_t` 作为 Critic 回归目标
- **PPO-clip loss**：`-min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)`，限制策略更新幅度
- **KL 惩罚**：构造成 per-token reward，均摊到每个生成 token 上
- **总损失**：`L_clip + vf_coef·L_value + ent_coef·L_entropy`，一次 backward，Actor/Critic 各自 optimizer 更新

---

## ⚠️ 踩坑提示

1. **模型精度用 fp32，不要用 bf16**
   GPT2 在 bf16 下前向会因激活值离群点产生 `NaN`（生成时触发 multinomial CUDA 报错）；
   被训练的模型用 bf16 还有"更新量被精度截断"的陷阱。本项目 Actor/Critic/Ref/Reward 全部用 fp32。

2. **生成必须左填充**
   `tokenizer.padding_side = "left"`，否则 generate 会从 padding 后面续写。

3. **奖励模型 tokenizer 必须和 Actor 一致**
   换 HF 预训练 RM 时，要选 GPT2 词表的（如 `Ray2333/gpt2-*-reward_model`），
   否则 Actor 生成的 token id 喂给 RM 是乱码。

4. **显存紧张时**
   调小 `rlhf_trainer.py` 里的 `batch_size`、`config.ppo.max_new_tokens`、`rollout_batch_size`。

5. **奖励尺度影响 kl_coef**
   不同 RM 输出分数范围不同。若 `reward/mean` 数值远大于 KL 项，Actor 会无视 KL 约束，需调大 `kl_coef`。

---

## 📌 训练顺序速查

```
预训练 GPT2（现成）
      │
      ├─ 方式A: 直接用 HF 预训练 RM ──┐
      │                              ├──► python train_ppo.py ──► 对齐后的 Actor
      └─ 方式B: train_reward.py ──────┘
```


