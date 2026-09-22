# PPO-RLHF

A **RLHF (Reinforcement Learning with Human Feedback)** training framework built from scratch, using the PPO algorithm to align language models.
The code uses GPT-2 as an example.

---

## RLHF Three-Stage

```
Phase 1: SFT (Supervised Fine-Tuning) —— Completed outside this project; use the pre-trained GPT-2 directly as the starting point
Phase 2: Training the Reward Model —— train_reward.py (You can also use the pre-trained RM from HF directly)
Phase 3: PPO Reinforcement Learning —— train_ppo.py
```

**4 models**:

| model | train or not | function |
|------|---------|------|
| **Actor** | yes | The optimized policy network is responsible for generating responses. |
| **Critic** | yes | Value network, estimating state value V(s), used for GAE |
| **Reference** | no | A copy of the SFT model; the KL penalty is calculated to prevent the Actor from deviating too far |
| **Reward Model** | no | Rate the generated response |

total reward:

$R = \mathbb{E}_{y \sim \pi_\theta(y|x)}[r(x, y)] - \beta \cdot \mathbb{D}_{KL}[\pi_{\theta}(y|x) || \pi_{ref}(y|x)]$

---

## Project structure

```
PPO_RLHF/
├── config.py          # Centralized Management of All Hyperparameters (dataclass)
├── model.py           # Definitions of the Actor, Critic, and Reference models
├── reward_model.py    # Reward model (self-training version + HF pre-training version), preference loss
├── data.py            # Preference dataset, Prompt dataset, DataLoader
├── ppo_trainer.py     # PPO core algorithm: rollout, GAE, PPO-clip loss, update
├── rlhf_trainer.py    # Training workflow coordinator: links all modules into a main loop
├── utils.py           # Utility functions: seed, logger, checkpoint, KL, masked_mean
├── train_reward.py    # Entry point 1: Train the reward model
└── train_ppo.py       # Entry point 2: PPO RLHF main training
```

---

## Environment Dependency

```bash
pip install torch transformers datasets tensorboard tqdm numpy
```

- Python 3.10+
- A CUDA-compatible GPU (recommended video memory $\gt$ 8 GB)

---

## Quick Start

### Method A: Use a pre-trained reward model from HuggingFace (recommended; skip Step 2)

`config.py` A pre-trained RM based on the GPT-2 vocabulary is configured by default:

```python
reward_model_name = "Ray2333/gpt2-large-harmless-reward_model"
```

Run PPO directly:

```bash
python train_ppo.py
```

> This RM uses the GPT-2 tokenizer and shares a vocabulary with the Actor, eliminating the need for any tokenizer conversion.

### Method B: Train the reward model yourself

```bash
# 1. First, train the reward model (the Anthropic/hh-rlhf dataset will be downloaded automatically).
python train_reward.py
# Output: checkpoints/reward_model/reward_model.pt

# 2. Run PPO again (you’ll need to change `rlhf_trainer.py` back to load the self-trained RM)
python train_ppo.py
```

### Monitoring Training

```bash
tensorboard --logdir ./logs
```

Key metrics:
- `reward/mean` —— average reward; should **increase** over training
- `kl/mean` —— how far the Actor drifts from the SFT model; **should not explode**
- `loss/actor` `loss/critic` `loss/entropy` —— individual loss terms

---

## Key Configuration (config.py)

| Parameter | Default | Description |
|------|------|------|
| `ppo.actor_lr` | 1e-6 | Actor learning rate (must be very small in the RL phase) |
| `ppo.critic_lr` | 1e-5 | Critic learning rate; can be slightly larger than the Actor's |
| `ppo.clip_eps` | 0.2 | PPO-clip range; smaller means more conservative updates |
| `ppo.gamma` | 0.99 | Discount factor |
| `ppo.gae_lambda` | 0.95 | GAE λ, trades off bias vs. variance |
| `ppo.kl_coef` | 0.1 | KL penalty coefficient; larger keeps the policy closer to SFT |
| `ppo.ent_coef` | 0.01 | Entropy regularization, encourages exploration |
| `ppo.ppo_epochs` | 4 | Number of passes over each batch of rollout data |
| `ppo.total_steps` | 1000 | Total training steps |
| `ppo.max_new_tokens` | 128 | Maximum length of generated responses |

---

## Algorithm Highlights

- **Rollout**: the Actor generates responses → record log-probs, reference-policy log-probs, rewards, and value estimates
- **GAE**: compute advantages backward via $A_t = \delta_t + \gamma\lambda·A_{t+1}$; use $G_t = A_t + V_t$ as the Critic's regression target
- **PPO-clip loss**: $-min(r_t \cdot A_t,\ clip(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t)$, which bounds the size of each policy update
- **KL penalty**: built as a per-token reward, spread over every generated token
- **Total loss**: $L_{clip} + vf_{coef} \cdot L_{value} + ent_{coef} \cdot L_{entropy}$, one backward pass, with separate optimizers updating the Actor and Critic

---

## Pitfalls

1. **Use fp32, not bf16**
   In bf16, GPT-2's forward pass can produce `NaN` due to activation outliers (triggering a multinomial CUDA error during generation);
   training a model in bf16 also risks updates being truncated by the limited precision. This project uses fp32 for Actor/Critic/Ref/Reward.

2. **Generation requires left padding**
   Set `tokenizer.padding_side = "left"`; otherwise generate will continue from after the padding.

3. **The reward model's tokenizer must match the Actor's**
   When switching to an HF pre-trained RM, pick one with the GPT-2 vocabulary (e.g. `Ray2333/gpt2-*-reward_model`);
   otherwise the token ids generated by the Actor are gibberish to the RM.

4. **When GPU memory is tight**
   Reduce `batch_size` in `rlhf_trainer.py`, `config.ppo.max_new_tokens`, and `rollout_batch_size`.

5. **Reward scale affects kl_coef**
   Different RMs output scores in different ranges. If `reward/mean` is much larger than the KL term, the Actor will ignore the KL constraint; increase `kl_coef`.

---

## Training Order Cheat Sheet

```
Pre-trained GPT-2 (off the shelf)
      │
      ├─ Method A: HF pre-trained RM ──┐
      │                                ├──► python train_ppo.py ──► Aligned Actor
      └─ Method B: train_reward.py ────┘
```


