# =============================================================================
# reward_model.py —— 奖励模型的架构定义与训练损失
#
# 奖励模型（RM）的作用：
#   给一个 (prompt, response) 对打分，分数越高代表人类越偏好这个回复
#
# 训练数据格式：
#   人类标注的偏好对：(prompt, chosen_response, rejected_response)
#   chosen 是人类更喜欢的回复，rejected 是人类不喜欢的回复
#
# 训练目标（Bradley-Terry 偏好模型）：
#   最大化 log σ(r_chosen - r_rejected)
#   等价于最小化 -log σ(r_chosen - r_rejected)
#   其中 σ 是 sigmoid 函数，r 是奖励模型打的分数
#
# 直觉：让 chosen 的分数比 rejected 的分数更高
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification
from config import RewardModelConfig


class PretrainedRewardModel(nn.Module):
    """
    直接加载 HuggingFace 上预训练好的 reward model（无需自己训练）。

    适用：基于 GPT2 词表的 reward model，例如
      - Ray2333/gpt2-large-harmless-reward_model
      - Ray2333/gpt2-large-helpful-reward_model
    这类模型用 GPT2 tokenizer，和你的 actor 共享同一套词表，
    所以 collect_rollouts 里可以直接把 actor 生成的 token id 喂进来打分，
    不需要任何 tokenizer 转换。

    对外接口和自训练的 RewardModel 完全一致：
      forward(input_ids, attention_mask) -> (B,)
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        # AutoModelForSequenceClassification：输出 (B, num_labels) 的 logits
        # 这类 reward model 是 num_labels=1，logit 本身就是 reward 分数
        # 注意：GPT2 系列在 bf16 下前向会产生 NaN（激活值离群点），
        # reward 算出来就是 nan，会污染整个训练。所以这里用 fp32。
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.reward_model_name,
            num_labels=1,
        )
        # GPT2 没有 pad_token，SequenceClassification 需要 pad_token_id
        # 来定位"最后一个有效 token"，否则会取到 padding 位置
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # 冻结参数（reward model 在 PPO 阶段不训练）
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        """
        返回每条序列的 reward 分数 (B,)

        注意：传入的 input_ids 必须是 GPT2 词表编码的（actor 生成的），
        和本 RM 的词表一致，才能正确打分。
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.squeeze(-1)  # (B, 1) -> (B,)


class RewardModel(nn.Module):
    """
    奖励模型：backbone + 线性打分头

    结构和 Critic 类似：
      backbone（预训练语言模型）→ 取最后 token 的 hidden state → 线性层 → 标量分数

    为什么取最后 token？
      对于 causal LM，最后一个非 padding token 包含了整个序列的信息
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        # TODO: 加载预训练 backbone
        # 提示：用 AutoModel.from_pretrained(config.reward_model_name)
        #       如果显存不够，可以用 8bit 量化：load_in_8bit=True
        self.backbone = AutoModel.from_pretrained(config.reward_model_name)

        # TODO: 定义打分头（Score Head）
        # 输入：backbone 的 hidden size
        # 输出：1（一个标量分数）
        # 提示：nn.Linear(hidden_size, 1)
        hidden_size = self.backbone.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播：对给定的 (prompt + response) 序列输出一个奖励分数

        关键点：取序列中 最后一个有效 token（非 padding）的 hidden state

        步骤：
          1. 通过 backbone 得到所有 token 的 hidden states: (batch, seq_len, hidden_size)
          2. 找到每条序列最后一个有效 token 的位置
             提示：attention_mask.sum(dim=-1) - 1 就是最后有效 token 的下标
          3. 用 gather 或 index 操作取出该位置的 hidden state: (batch, hidden_size)
          4. 通过 score_head 得到分数: (batch, 1)
          5. squeeze 成 (batch,)

        返回：
          scores: (batch,) —— 每条序列的奖励分数

        TODO: 实现上述步骤
        """
        output = self.backbone(input_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
        last_hidden: torch.Tensor = output.hidden_states[-1]  # 最后一层的(B, S, H)
        # 找到最后一个有效token的位置
        last_token_indices = attention_mask.sum(dim=-1) - 1  # (B,)
        
        # 最后一个有效token的向量表示
        H = last_hidden.size(-1)
        idx = last_token_indices.view(-1, 1, 1).expand(-1, 1, H)  # (B, 1, H)
        last_token_hidden = last_hidden.gather(dim=1, index=idx).squeeze(1)  # (B, H)
        score = self.score_head(last_token_hidden)  # (B, 1)
        
        return score.squeeze(-1)  # (B,)
        


def compute_reward_loss(reward_model: RewardModel, chosen_ids, rejected_ids,
                        chosen_mask, rejected_mask):
    """
    计算 Bradley-Terry 偏好损失

    数学公式：
      loss = -log σ(r_chosen - r_rejected)
           = -log sigmoid(score_chosen - score_rejected)

    参数：
      chosen_ids:   (batch, seq_len) —— chosen 序列的 token ids
      rejected_ids: (batch, seq_len) —— rejected 序列的 token ids
      chosen_mask:  (batch, seq_len) —— chosen 序列的 attention mask
      rejected_mask:(batch, seq_len) —— rejected 序列的 attention mask

    步骤：
      1. 分别对 chosen 和 rejected 过 reward_model，得到两组分数
      2. 计算 score_chosen - score_rejected
      3. 对差值取 sigmoid，然后取 log，再取负（最小化负对数似然）
         等价于：F.logsigmoid(score_chosen - score_rejected).mean().neg()

    返回：
      loss: 标量 —— 偏好学习损失
      accuracy: 标量 —— chosen 分数 > rejected 分数的比例（监控指标）

    TODO: 实现上述步骤
    提示：accuracy = (score_chosen > score_rejected).float().mean()
    """
    score_chosen = reward_model(chosen_ids, attention_mask=chosen_mask)  # (B,)
    score_rejected = reward_model(rejected_ids, attention_mask=rejected_mask)  # (B,)
    
    delta = score_chosen - score_rejected
    loss = -F.logsigmoid(delta).mean()
    
    accuracy = (score_chosen > score_rejected).sum() / score_chosen.size(0)
    
    return loss, accuracy


def train_reward_model_step(reward_model, optimizer, batch, device):
    """
    奖励模型的单步训练（一个 batch 的前向 + 反向 + 更新）

    参数：
      batch: 字典，包含 chosen_ids, rejected_ids, chosen_mask, rejected_mask

    步骤：
      1. 把 batch 数据移到 device
      2. 调用 compute_reward_loss 得到 loss 和 accuracy
      3. optimizer.zero_grad()
      4. loss.backward()
      5. （可选）梯度裁剪：nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
      6. optimizer.step()

    返回：
      loss.item(), accuracy.item()

    TODO: 实现上述步骤
    """
    reward_model.train()
    
    chosen_ids = batch['chosen_ids'].to(device)
    rejected_ids = batch['rejected_ids'].to(device)
    chosen_mask = batch['chosen_mask'].to(device)
    rejected_mask = batch['rejected_mask'].to(device)
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss, accuracy = compute_reward_loss(reward_model, chosen_ids,
                                             rejected_ids, chosen_mask, 
                                             rejected_mask)
    
    loss.backward()
    nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item(), accuracy.item()


# if __name__ == "__main__":
#     import torch
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification

#     rm_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-harmless-reward_model')
#     reward_model = AutoModelForSequenceClassification.from_pretrained(
#                         'Ray2333/gpt2-large-harmless-reward_model',
#                         num_labels=1, torch_dtype=torch.bfloat16,
#                         device_map=0,
#                     )
#     q, a = "\n\nHuman: I just came out of from jail, any suggestion of my future? \n\nAssistant:", "I'm so happy that you are ou of jail. First of all you can go search a job and intergrate the society."
#     inputs = rm_tokenizer(q, a, return_tensors='pt', truncation=True)
#     with torch.no_grad():
#         reward = reward_model(**(inputs.to(0))).logits[0].cpu().detach().item()
#         print(reward)
