# =============================================================================
# data.py —— 数据集定义与 DataLoader 构建
#
# RLHF 需要两类数据：
#
# 1. 偏好数据集（用于训练奖励模型）
#    格式：(prompt, chosen_response, rejected_response)
#    来源：人类标注，或 AI 生成后人类排序
#    示例数据集：Anthropic/hh-rlhf, OpenAI summarize-from-feedback
#
# 2. Prompt 数据集（用于 PPO rollout）
#    格式：只有 prompt，没有 response
#    PPO 过程中，Actor 自己生成 response，奖励模型给分
#    来源：可以是偏好数据集里的 prompt 部分，或其他指令数据集
# =============================================================================

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import List, Dict
import torch
import json


class PreferenceDataset(Dataset):
    """
    偏好数据集：用于训练奖励模型

    每条数据包含：
      - prompt: 输入提示
      - chosen: 人类更偏好的回复
      - rejected: 人类不偏好的回复

    __getitem__ 返回 tokenize 后的结果，供 DataLoader 使用
    """

    def __init__(self, data: List[Dict], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        参数：
          data: 列表，每个元素是 {"prompt": str, "chosen": str, "rejected": str}
          tokenizer: 用于编码文本
          max_length: 最大序列长度（超出则截断）

        TODO:
          1. 保存 data, tokenizer, max_length
          2. 这里可以做数据过滤（过长的丢弃，或确保 chosen != rejected）
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # TODO: 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回第 idx 条数据的 tokenize 结果

        步骤：
          1. 取出 prompt, chosen, rejected
          2. 拼接：chosen_text = prompt + chosen，rejected_text = prompt + rejected
          3. 分别 tokenize，设置 max_length, padding="max_length", truncation=True
          4. 返回字典：
             {
               "chosen_ids":    tensor, shape (max_length,)
               "chosen_mask":   tensor, shape (max_length,)
               "rejected_ids":  tensor, shape (max_length,)
               "rejected_mask": tensor, shape (max_length,)
             }

        TODO: 实现上述步骤
        提示：tokenizer(text, return_tensors="pt", ...) 返回结果记得 squeeze(0) 去掉 batch 维
        """
        prompt = self.data[idx]['prompt']
        chosen = self.data[idx]['chosen']
        rejected = self.data[idx]['rejected']
        
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        
        tokenized_chosen = self.tokenizer(chosen_text, return_tensors='pt',
                                          max_length=self.max_length,
                                          truncation=True, padding='max_length')
        chosen_ids = tokenized_chosen['input_ids'].squeeze(0)
        chosen_mask = tokenized_chosen['attention_mask'].squeeze(0)
        
        tokenized_rejected = self.tokenizer(rejected_text, return_tensors='pt',
                                          max_length=self.max_length,
                                          truncation=True, padding='max_length')
        rejected_ids = tokenized_rejected['input_ids'].squeeze(0)
        rejected_mask = tokenized_rejected['attention_mask'].squeeze(0)
        
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask
        }


class PromptDataset(Dataset):
    """
    Prompt 数据集：用于 PPO rollout 阶段

    只包含 prompt，不包含 response。
    PPO 训练时，Actor 根据 prompt 生成 response，再由奖励模型打分。
    """

    def __init__(self, prompts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 256):
        """
        参数：
          prompts: prompt 字符串列表
          tokenizer: 编码器
          max_length: prompt 最大长度（太长会挤占生成空间）

        TODO: 保存参数
        """
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.max_length = max_length

    def __len__(self):
        # TODO: 返回 prompt 数量
        return len(self.prompts)

    def __getitem__(self, idx):
        """
        返回第 idx 个 prompt 的 tokenize 结果

        步骤：
          1. 取出 prompt 字符串
          2. tokenize，注意：
             - padding_side 应该是 "left"（生成时需要左填充）
             - truncation=True, max_length=self.max_length
          3. 返回字典：
             {
               "input_ids":      tensor, shape (max_length,)
               "attention_mask": tensor, shape (max_length,)
               "prompt_text":    str（原始字符串，方便调试）
             }

        TODO: 实现上述步骤
        """
        prompt = self.prompts[idx]
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt',
                                          max_length=self.max_length,
                                          truncation=True, padding='max_length')
        
        return {
            "input_ids": tokenized_prompt['input_ids'].squeeze(0),
            "attention_mask": tokenized_prompt['attention_mask'].squeeze(0),
            "prompt_text": prompt
        }


def load_preference_data(data_path: str) -> List[Dict]:
    """
    从文件加载偏好数据

    你需要根据实际数据格式来实现。
    常见格式：
      - JSON Lines (.jsonl)：每行一个 JSON 对象
      - HuggingFace datasets：用 datasets.load_dataset() 加载

    TODO: 实现数据加载逻辑
    示例（jsonl 格式）：
      with open(data_path) as f:
          return [json.loads(line) for line in f]

    如果用 HuggingFace datasets：
      from datasets import load_dataset
      ds = load_dataset("Anthropic/hh-rlhf")
      # 转换成 {"prompt": ..., "chosen": ..., "rejected": ...} 格式
    """
    def _extract_prompt(chosen: str) -> str:
        # 找最后一个 "Assistant:" 出现的位置
        last_assistant = chosen.rfind("\n\nAssistant:")
        if last_assistant == -1:
            return chosen
        # 截取到 "Assistant:" 结束（含这个标记，让模型知道该生成回复了）
        return chosen[:last_assistant + len("\n\nAssistant:")]
        
    if data_path.endswith(".jsonl"):
        with open(data_path) as f:
            rows = [json.loads(line) for line in f]
        for row in rows:
            if "prompt" not in row:
                row["prompt"] = _extract_prompt(row['chosen'])
                row["chosen"] = row["chosen"][len(row["prompt"]):]
                row["rejected"] = row["rejected"][len(row["prompt"]):]
        return rows
    else:
        try:
            ds = load_dataset(data_path)
            res = []
            for row in ds['train']:
                chosen = row['chosen']
                rejected = row['rejected']
                chosen_prompt = _extract_prompt(chosen)
                rejected_prompt = _extract_prompt(rejected)
                
                chosen_response = chosen[len(chosen_prompt):]
                rejected_response = rejected[len(rejected_prompt):]
                res.append({
                    "prompt": chosen_prompt, 
                    "chosen": chosen_response, 
                    "rejected": rejected_response
                    })
            return res
        except Exception as e:
            print(f"Can't load dataset from path : {data_path}, error : {e}")
            raise


def load_prompt_data(data_path: str) -> List[str]:
    """
    从文件加载 prompt 列表

    TODO: 实现 prompt 加载
    可以从偏好数据里提取 prompt：
      data = load_preference_data(data_path)
      return [d["prompt"] for d in data]
    """
    data = load_preference_data(data_path)
    return [d["prompt"] for d in data]


def get_preference_dataloader(data: List[Dict], tokenizer, max_length: int,
                              batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    构建奖励模型训练用的 DataLoader

    TODO:
      1. 构建 PreferenceDataset
      2. 构建 DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
      3. 返回 DataLoader
    """
    preference_ds = PreferenceDataset(data, tokenizer=tokenizer, max_length=max_length)
    preference_dl = DataLoader(preference_ds, batch_size=batch_size, shuffle=shuffle,
                               num_workers=4, pin_memory=True)
    return preference_dl


def get_prompt_dataloader(prompts: List[str], tokenizer, max_length: int,
                          batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    构建 PPO rollout 用的 DataLoader

    TODO: 同上，用 PromptDataset 和 DataLoader
    """
    prompt_ds = PromptDataset(prompts, tokenizer=tokenizer, max_length=max_length)
    prompt_dl = DataLoader(prompt_ds, batch_size=batch_size, shuffle=shuffle)
    return prompt_dl


# if __name__ == "__main__":
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
#     train_data = load_preference_data("Anthropic/hh-rlhf")
#     lengths = [len(tokenizer(d["prompt"] + d["chosen"])["input_ids"]) for d in train_data[:500]]
#     print(f"超过512的比例: {sum(l > 512 for l in lengths) / len(lengths):.2%}")
#     print(f"平均长度: {sum(lengths)/len(lengths):.0f}, 最大长度: {max(lengths)}")
