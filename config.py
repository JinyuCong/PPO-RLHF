# =============================================================================
# config.py —— 所有超参数的集中管理
# 好处：改参数只需改这一个文件，不用翻代码
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    语言模型相关配置
    这里以 HuggingFace 模型为例，你需要填入：
      - 预训练模型的路径或名称（如 "gpt2" / "Qwen/Qwen2-0.5B"）
      - 模型最大序列长度
    """
    # TODO: 填入你想用的预训练模型名称或本地路径
    model_name: str = "gpt2"

    # TODO: 模型最大接受的 token 序列长度
    max_length: int = 256

    # TODO: 词表大小（通常从 tokenizer 自动读取，这里也可以显式写）
    vocab_size: Optional[int] = None


@dataclass
class RewardModelConfig:
    """
    奖励模型专属配置
    奖励模型通常和 Actor 用同一个 backbone，但最后一层换成输出单个标量
    """
    # TODO: 奖励模型用的预训练 backbone（可以和 ModelConfig.model_name 相同）
    reward_model_name: str = "gpt2"

    # TODO: 奖励模型 checkpoint 保存路径
    reward_model_save_path: str = "./checkpoints/reward_model"

    # TODO: 训练奖励模型时的学习率
    reward_lr: float = 1e-5

    # TODO: 奖励模型训练的 epoch 数
    reward_epochs: int = 3

    # TODO: 奖励模型训练的 batch size
    reward_batch_size: int = 16


@dataclass
class PPOConfig:
    """
    PPO 算法核心超参数
    理解每个参数的含义是写好 PPO 的关键，注释里有解释
    """
    # --- 基础训练参数 ---
    # TODO: Actor（策略网络）的学习率，通常比监督学习小一个数量级
    actor_lr: float = 1e-6

    # TODO: Critic（价值网络）的学习率，可以比 actor 稍大
    critic_lr: float = 1e-5

    # TODO: 每次 PPO 更新时用的 mini-batch 大小
    ppo_batch_size: int = 4

    # TODO: 每批 rollout 数据上重复训练几遍（PPO 的 "epoch"）
    ppo_epochs: int = 4

    # --- PPO clip 参数 ---
    # clip_eps：限制策略更新幅度的核心参数
    # 更新比率 r_t = π_new(a|s) / π_old(a|s)
    # clip 后：min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
    # TODO: 通常取 0.1 ~ 0.2，值越小更新越保守
    clip_eps: float = 0.2

    # --- 广义优势估计 (GAE) 参数 ---
    # gamma：折扣因子，控制未来奖励的重要性
    # TODO: 通常取 0.99，接近 1 表示更重视长期回报
    gamma: float = 0.99

    # gae_lambda：GAE 的 λ 参数，平衡 bias 和 variance
    # λ=1 退化为 Monte Carlo，λ=0 退化为 TD(0)
    # TODO: 通常取 0.95
    gae_lambda: float = 0.95

    # --- 损失函数权重 ---
    # Critic 损失系数：总损失 = actor_loss + vf_coef * critic_loss - ent_coef * entropy
    # TODO: 通常取 0.1 ~ 1.0
    vf_coef: float = 0.1

    # 熵正则化系数：鼓励探索，防止策略过早收敛
    # TODO: 通常取 0.01，值越大越鼓励随机性
    ent_coef: float = 0.01

    # --- KL 散度惩罚 ---
    # KL 惩罚防止 Actor 偏离 SFT 模型太远（RLHF 的关键机制）
    # 总奖励 = reward_model_score - kl_coef * KL(π_new || π_ref)
    # TODO: 通常取 0.01 ~ 0.1，值越大越保守
    kl_coef: float = 0.05

    # --- 训练规模 ---
    # TODO: 总共进行多少轮 PPO 迭代（每轮 = 一次 rollout + 多次更新）
    total_steps: int = 1000

    # TODO: 每次 rollout 生成多少条回复
    rollout_batch_size: int = 16

    # TODO: 生成回复时的最大新 token 数
    max_new_tokens: int = 128


@dataclass
class TrainingConfig:
    """
    通用训练配置（设备、路径、日志等）
    """
    data_path: str = "Anthropic/hh-rlhf"
    
    # TODO: 训练设备，"cuda" / "cpu" / "mps"（苹果芯片）
    device: str = "cuda"

    # TODO: 日志输出目录（使用 tensorboard 或 wandb）
    log_dir: str = "./logs"

    # TODO: 模型 checkpoint 保存目录
    output_dir: str = "./checkpoints/ppo"

    # TODO: 每隔多少步保存一次 checkpoint
    save_steps: int = 100

    # TODO: 每隔多少步打印/记录一次日志
    log_steps: int = 10

    # TODO: 随机种子，保证可复现性
    seed: int = 42

    # TODO: 是否使用混合精度训练（fp16/bf16），可以加速并节省显存
    use_bf16: bool = False


@dataclass
class Config:
    """
    总配置入口：把所有子配置组合在一起
    使用方式：cfg = Config()，然后 cfg.model.model_name, cfg.ppo.clip_eps, etc.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardModelConfig = field(default_factory=RewardModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # TODO（可选）：你可以添加一个 from_yaml() 类方法，从 yaml 文件加载配置
    # 这样不需要改代码，只改 yaml 就能切换实验配置
