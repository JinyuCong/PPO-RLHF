# =============================================================================
# train_reward.py —— 第二阶段入口：训练奖励模型
#
# RLHF 三阶段：
#   阶段1: SFT（监督微调）—— 通常在此项目外完成
#   阶段2: 训练奖励模型  ← 这个脚本
#   阶段3: PPO 强化学习  ← train_ppo.py
#
# 运行方式：python train_reward.py
# =============================================================================

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config, RewardModelConfig
from reward_model import RewardModel, train_reward_model_step
from data import load_preference_data, get_preference_dataloader
from utils import set_seed, get_logger, print_model_info


def main():
    # ─── 1. 加载配置 ─────────────────────────────────────────────────────────
    config = Config()
    # TODO: 可以在这里覆盖默认配置，例如：
    #   config.reward.reward_epochs = 5
    #   config.training.device = "cuda:0"

    logger = get_logger("reward_training", config.training.log_dir)
    set_seed(config.training.seed)
    device = config.training.device

    # ─── 2. 加载 tokenizer ───────────────────────────────────────────────────
    # TODO: 用 AutoTokenizer.from_pretrained(config.reward.reward_model_name)
    #       注意设置 pad_token（GPT2 没有默认 pad_token）
    tokenizer = AutoTokenizer.from_pretrained(config.reward.reward_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ─── 3. 加载数据 ─────────────────────────────────────────────────────────
    # TODO: 调用 load_preference_data 加载你的偏好数据集
    #       数据格式：[{"prompt": str, "chosen": str, "rejected": str}, ...]
    logger.info("Loading preference data...")
    train_data = load_preference_data("Anthropic/hh-rlhf")
    # eval_data = ...（可选，用于验证集评估）

    # TODO: 构建 DataLoader
    train_loader = get_preference_dataloader(
        train_data, tokenizer,
        max_length=config.model.max_length,
        batch_size=config.reward.reward_batch_size,
        shuffle=True
    )

    # ─── 4. 初始化奖励模型 ───────────────────────────────────────────────────
    logger.info("Initializing reward model...")
    reward_model = RewardModel(config.reward).to(device)
    print_model_info(reward_model, "RewardModel")

    # ─── 5. 初始化优化器 ─────────────────────────────────────────────────────
    # TODO: AdamW，学习率用 config.reward.reward_lr
    optimizer = AdamW(reward_model.parameters(), lr=config.reward.reward_lr)

    # TODO（可选）：学习率调度器
    #   from transformers import get_linear_schedule_with_warmup
    #   scheduler = get_linear_schedule_with_warmup(optimizer, ...)

    # ─── 6. 主训练循环 ───────────────────────────────────────────────────────
    logger.info("Starting reward model training...")

    for epoch in range(config.reward.reward_epochs):
        reward_model.train()
        epoch_losses = []
        epoch_accs = []

        # TODO: 遍历 train_loader
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        
            loss, accuracy = train_reward_model_step(
                reward_model, optimizer, batch, device
            )
            epoch_losses.append(loss)
            epoch_accs.append(accuracy)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        avg_acc = sum(epoch_accs) / len(epoch_accs) if epoch_accs else 0
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={avg_acc:.4f}")

        # TODO（可选）：在验证集上评估

    # ─── 7. 保存模型 ─────────────────────────────────────────────────────────
    # TODO: 保存奖励模型的权重
    import os
    os.makedirs(config.reward.reward_model_save_path, exist_ok=True)
    torch.save(reward_model.state_dict(),
               f"{config.reward.reward_model_save_path}/reward_model.pt")
    logger.info(f"Reward model saved to {config.reward.reward_model_save_path}")


if __name__ == "__main__":
    main()
