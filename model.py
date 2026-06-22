# =============================================================================
# model.py —— 三个核心模型的定义
#
# RLHF 中同时存在三个模型：
#   1. Actor      —— 被训练的策略网络（语言模型），负责生成回复
#   2. Critic     —— 价值网络，估计当前状态的期望回报 V(s)
#   3. Reference  —— SFT 后冻结的参考模型，用于计算 KL 惩罚
#
# 三者通常共享同一个 backbone（节省显存），只是最后一层不同：
#   Actor   → LM head（输出词表维度的 logits）
#   Critic  → Value head（输出一个标量）
#   Reference → 和 Actor 结构相同，但参数冻结
# =============================================================================

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from config import ModelConfig


class ActorModel(nn.Module):
    """
    Actor（策略网络）：RLHF 中被优化的语言模型

    职责：
      - 接收 prompt，自回归地生成 token 序列（rollout）
      - 在 PPO 更新时，计算每个 token 的对数概率 log π(a_t | s_t)

    思路：
      直接包装一个 HuggingFace CausalLM，不需要魔改内部结构
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: 用 AutoModelForCausalLM.from_pretrained 加载预训练模型
        # 提示：加载时可以传入 torch_dtype=torch.bfloat16 节省显存
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, 
                                                          torch_dtype=torch.bfloat16)

        # TODO: 加载对应的 tokenizer
        # 注意：需要设置 padding_side="left"（因为生成时需要左填充）
        #       并确保 pad_token 存在（GPT2 没有 pad_token，需要手动设置）
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播：给定 input_ids，返回每个位置的 logits

        输入：
          input_ids:      (batch, seq_len) —— token ID 序列
          attention_mask: (batch, seq_len) —— 1=有效 token，0=padding

        返回：
          logits: (batch, seq_len, vocab_size) —— 每个位置的原始得分

        TODO: 直接调用 self.model(input_ids, attention_mask=attention_mask)
              返回 output.logits
        """
        output = self.model(input_ids, attention_mask=attention_mask)
        return output.logits  # (B, S, V)

    def get_log_probs(self, input_ids, attention_mask=None):
        """
        计算序列中每个 token 的对数概率（PPO 更新时需要）

        关键步骤：
          1. 调用 forward 得到 logits: (batch, seq_len, vocab_size)
          2. 对 logits 做 log_softmax，得到 log_probs: (batch, seq_len, vocab_size)
          3. 用 gather 取出实际生成 token 对应的 log_prob
             —— 注意：token t 的概率由 t-1 位置的 logits 预测（自回归）
                所以 logits[:, :-1] 对应 labels[:, 1:]

        返回：
          log_probs: (batch, seq_len-1) —— 每个生成 token 的 log 概率

        TODO: 实现上述三步
        """
        logits = self.forward(input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)  # (B, S-1, V)
        labels = input_ids[:, 1:].unsqueeze(-1)  # (B, S-1)
        p_token = log_probs.gather(dim=-1, index=labels).squeeze(-1)
        return p_token

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=128, **kwargs):
        """
        生成回复（rollout 阶段使用）

        TODO: 调用 self.model.generate(...)
        提示：
          - do_sample=True 启用随机采样（训练时需要探索）
          - temperature 控制随机性
          - 返回完整序列（包含 prompt 部分）
        """
        output = self.model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0
            )
        return output


class CriticModel(nn.Module):
    """
    Critic（价值网络）：估计每个状态的期望累计回报 V(s_t)

    职责：
      - 给定已生成的 token 序列，为每个位置输出一个标量值估计
      - 用于计算 GAE（广义优势估计）中的 baseline

    结构：
      backbone（同 Actor）+ 线性层（hidden_size → 1）

    注意：Critic 通常和 Actor 共享底层 backbone，但在实现上可以独立，
          取决于你的显存预算。简单起见，这里用独立 Critic。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: 加载预训练模型作为 backbone
        # 提示：用 AutoModel.from_pretrained（不带 LM head）更干净
        #       或者直接用 CausalLM 然后取 hidden states
        self.backbone = AutoModel.from_pretrained(config.model_name, 
                                                             torch_dtype=torch.bfloat16)

        # TODO: 定义 value head
        # 输入维度 = 模型 hidden size，输出维度 = 1
        # 提示：从 self.backbone.config.hidden_size 获取 hidden size
        hidden_size = self.backbone.config.hidden_size  # 768
        self.value_head = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播：返回每个 token 位置的状态价值估计

        步骤：
          1. 通过 backbone 得到 hidden states: (batch, seq_len, hidden_size)
          2. 通过 value_head 压缩到: (batch, seq_len, 1)
          3. squeeze 最后一维，得到: (batch, seq_len)

        返回：
          values: (batch, seq_len) —— 每个位置的 V(s_t) 估计

        TODO: 实现上述步骤
        提示：需要在 backbone 调用时设置 output_hidden_states=True
              取最后一层 hidden state
        """
        output = self.backbone(input_ids, 
                               attention_mask=attention_mask,
                               output_hidden_states=True)
        last_hidden = output.hidden_states[-1]  # (B, S, H)
        return self.value_head(last_hidden).squeeze(-1)  # (B, S)
    

class ReferenceModel(nn.Module):
    """
    Reference（参考模型）：SFT 后的语言模型，参数完全冻结

    职责：
      - 计算参考策略下的 log 概率：log π_ref(a_t | s_t)
      - 用于 KL 惩罚：KL(π_actor || π_ref) 防止 Actor 偏离太远

    注意：这个模型 永远不更新梯度，所有操作都在 @torch.no_grad() 下进行
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: 加载和 Actor 相同的预训练模型
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name,
                                                          torch_dtype=torch.bfloat16)

        # TODO: 冻结所有参数（两种方式选一种）：
        #   方式1：for p in self.model.parameters(): p.requires_grad = False
        #   方式2：self.model = self.model.eval()  （eval 模式，但不冻结参数）
        #   推荐方式1，更彻底
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_log_probs(self, input_ids, attention_mask=None):
        """
        计算参考策略下每个 token 的对数概率

        实现和 ActorModel.get_log_probs 完全相同，但包裹在 @torch.no_grad() 中

        TODO: 复用 ActorModel.get_log_probs 的逻辑
        """
        model_out = self.model(input_ids, attention_mask=attention_mask)
        logits = model_out.logits  # (B, S-1, V)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, S-1, V)
        labels = input_ids[:, 1:].unsqueeze(-1)  # (B, S-1, 1)
        p_token = log_probs.gather(dim=-1, index=labels).squeeze(-1)  # (B, S-1)
        return p_token
        
        
def load_models(config: ModelConfig, device: str):
    """
    统一加载三个模型并移动到指定设备

    返回：(actor, critic, reference) 三元组

    TODO:
      1. 实例化 ActorModel, CriticModel, ReferenceModel
      2. 把三个模型都 .to(device)
      3. 把 reference 设为 eval 模式（不影响 actor/critic 的训练模式）
    """
    actor = ActorModel(config).to(device)
    critic = CriticModel(config).to(device)
    reference = ReferenceModel(config).to(device).eval()
    return actor, critic, reference

    