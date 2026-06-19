# =============================================================================
# train_ppo.py —— 第三阶段入口：PPO RLHF 主训练
#
# 前提：奖励模型已经训练完成（运行过 train_reward.py）
#
# 运行方式：python train_ppo.py
# =============================================================================

import torch
from config import Config
from rlhf_trainer import RLHFTrainer
from utils import set_seed, get_logger


def main():
    # ─── 1. 加载配置 ─────────────────────────────────────────────────────────
    config = Config()

    # TODO: 在这里覆盖你需要调整的配置，例如：
    #   config.ppo.kl_coef = 0.05          # KL 惩罚系数
    #   config.ppo.clip_eps = 0.2           # PPO clip 范围
    #   config.ppo.total_steps = 1000       # 总训练步数
    #   config.training.device = "cuda:0"   # 训练设备

    logger = get_logger("ppo_training", config.training.log_dir)
    logger.info("Config loaded:")
    logger.info(f"  actor_lr={config.ppo.actor_lr}")
    logger.info(f"  kl_coef={config.ppo.kl_coef}")
    logger.info(f"  total_steps={config.ppo.total_steps}")

    # ─── 2. 初始化训练器 ─────────────────────────────────────────────────────
    # RLHFTrainer 内部会完成：
    #   - 加载 actor, critic, ref_model, reward_model
    #   - 初始化优化器和数据加载器
    #   - 设置日志和 checkpoint
    logger.info("Initializing RLHF trainer...")
    trainer = RLHFTrainer(config)

    # ─── 3. （可选）从断点恢复 ────────────────────────────────────────────────
    # TODO: 如果有已有的 checkpoint，取消下面的注释：
    # checkpoint_path = "./checkpoints/ppo/step_500"
    # trainer.load(checkpoint_path)
    # logger.info(f"Resumed from {checkpoint_path}, step={trainer.global_step}")

    # ─── 4. 开始训练 ─────────────────────────────────────────────────────────
    logger.info("Starting PPO RLHF training...")
    logger.info("Monitor training with: tensorboard --logdir ./logs")
    trainer.train()

    # ─── 5. 训练结束，保存最终模型 ───────────────────────────────────────────
    logger.info("Training complete. Saving final model...")
    trainer.save()
    logger.info(f"Final model saved to {config.training.output_dir}")


if __name__ == "__main__":
    main()
