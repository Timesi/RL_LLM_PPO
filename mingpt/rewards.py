import torch
from torch import nn

from mingpt.model import Transformer


def calculate_advantage_and_returns(rewards, values, gamma, lambd):
    """Calculate the GAE estimate of the advantage."""
    lastgaelam = 0      # 存储上一个时间步的优势估计值
    advantages = torch.zeros_like(rewards)  # 存储优势估计值的张量
    gen_len = rewards.size(1)   # 时间步数

    for t in reversed(range(gen_len)):      # 从最后一步向前计算
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0       # 获取下一时刻价值
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]       # TD误差
        advantages[:, t] = lastgaelam = delta + gamma * lambd * lastgaelam  # GAE递推公式
    returns = advantages + values   # 回报 = 优势 + 状态价值

    return advantages, returns


def shift_completions_and_mask(prompt, completion, end_of_text, block_size):
    """
    This method processes the prompts and completions by stacking and padding them on the right.

    Steps:
    1. Extract the first non-padding token from the left-padded prompts.
    2. Identify the last token in the completions, truncating at the first end-of-text token.
    3. Concatenate the prompts and completions, right-padding the results with end-of-text tokens.
    4. Create attention masks and action masks for the concatenated sequences.

    Args:
        prompt (torch.Tensor): Tensor containing the prompt tokens, left-padded.
        completion (torch.Tensor): Tensor containing the completion tokens.
        end_of_text (int): Token ID representing the end-of-text.

    Returns:
        completions (torch.Tensor): Concatenated and padded prompt and completion sequences.
        attn_mask (torch.Tensor): Attention mask indicating which tokens should be attended to.
        action_mask (torch.Tensor): Action mask indicating which tokens are part of the completion.
    """
    # Get the first prompt token since they're left padded
    prompt_padding = (prompt != end_of_text).long().argmax(dim=1)

    # Get the last completion token
    token_ends = (completion == end_of_text).long().argmax(dim=1)

    # If there isn't a last completion token, select the end
    token_ends[torch.logical_and(token_ends == 0, completion[:, 0] != end_of_text)] = completion.size(1)

    prompt_size = prompt.size(1)

    # cat the prompt and completions and right pad the results
    # creating masks for the attn and the completion parts separately
    completions = []
    action_masks = []
    attn_masks = []
    for i, (prompt_start, completion_end) in enumerate(zip(prompt_padding, token_ends)):
        x = prompt[i, prompt_start:]
        c = completion[i,:completion_end]

        # we pad eot tokens to block_size + 1
        pad_size = block_size - prompt_size + prompt_start.item() - completion_end.item() + 1
        padding = torch.full((pad_size,), end_of_text, dtype=prompt.dtype, device=prompt.device)
        comp = torch.cat((x, c, padding))
        completions.append(comp)

        # true for values that are ignored in attention, opposite for the completion
        attn_mask = torch.full((block_size,), False, dtype=bool, device=prompt.device)
        if pad_size > 2:
            attn_mask[-pad_size+2:] = True # include the first eot token
        attn_masks.append(attn_mask)

        action_mask = attn_mask.clone()
        action_mask[:x.size(0)] = True
        action_mask = torch.logical_not(action_mask)
        action_masks.append(action_mask)

    action_mask = torch.stack(action_masks)
    attn_mask = torch.stack(attn_masks)
    completions = torch.stack(completions)
    target = completions[:, 1:].clone()

    return completions[:, :-1], attn_mask, action_mask, target


class ValueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = Transformer(config)
        self.prediction_head = nn.Linear(config.n_embd, 1, bias=True)
        torch.nn.init.normal_(self.prediction_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.prediction_head.bias)

    def forward(self, toks, attn_mask=None):
        x = self.transformer(toks, attention_mask=attn_mask) # (b, t, n_embd)
        rewards = self.prediction_head(x).squeeze(-1) # (b,)

        return rewards

# The reward model looks more complicated, but we are just batching positive and
# negative responses together and calculating the loss.
# We also gather the reward prediction from the last non-masked token.
class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = Transformer(config)      # 使用transformer作为特征提取模块
        self.prediction_head = nn.Linear(config.n_embd, 1)      # 预测头，一个线性层，将transformer的输出映射到标量奖励值
        torch.nn.init.normal_(self.prediction_head.weight, mean=0.0, std=0.02)      # 预测头权重使用小方差正态分布初始化
        torch.nn.init.zeros_(self.prediction_head.bias)     # 偏置初始化为0

    def forward(self, toks, attn_mask=None, positive_tokens=None, positive_mask=None):
        if positive_tokens is not None:
            assert toks.size(0) == positive_tokens.size(0)
            toks = torch.cat((toks, positive_tokens))

        if positive_mask is not None:
            assert attn_mask is not None
            attn_mask = torch.cat((attn_mask, positive_mask))

        # 使用Transformer处理输入文本序列
        x = self.transformer(toks, attention_mask=attn_mask) # (b, t, n_embd)=(批次大小，序列长度，嵌入维度)

        # 选择序列中最后一个非填充位置的表示作为整体表示，这是因为整个序列的质量应该由完整序列来评估
        if attn_mask is None:   # 判断是都存在掩码
            reward_idxs = -1    # 没有掩码，则使用序列的最后一个位置的表示作为整体表示
        else:
            # Gets the last non-masked value
            # 如果存在掩码，需要找到最后一个非掩码位置
            # （1）获取序列长度
            # attn_mask.size(1)： 获取序列的长度
            # （2）翻转掩码
            # torch.flip(attn_mask, dims=[1])： 沿维度1（序列维度）翻转掩码
            # 原始掩码：[False, False, False, True, True] （False表示有效token，True表示填充）
            # 翻转后：[True, True, False, False, False]
            # （3）转换数据类型并找到最小值索引
            # .to(torch.int64): 将布尔值转换为整数（False→0, True→1），argmin(dim=1): 找到每行中最小值的索引
            # 对于翻转后的掩码 [1, 1, 0, 0, 0]，最小值0的索引是2
            # （4） 计算原始序列中的位置
            # 假设序列长度为5，原始掩码为 [False, False, False, True, True]
            # 翻转后：[True, True, False, False, False] → [1, 1, 0, 0, 0]
            # argmin(dim=1) 返回2（第一个0的索引）
            # 计算：5 - 2 - 1 = 2
            # 位置2正好是最后一个非掩码位置
            reward_idxs = attn_mask.size(1) - torch.flip(attn_mask, dims=[1]).to(torch.int64).argmin(dim=1) - 1

        # 从x中取出每个batch中，样本的最后一个非填充位置
        # 假设 x 形状为 [3, 5, 768]（3个样本，每个序列5个token，每个token 768维）
        # torch.arange(x.size(0)) = [0, 1, 2]
        # reward_idxs = [2, 4, 1]
        # 索引操作会选出：
        # x[0, 2]：第0个样本的第2个位置的表示（768维）
        # x[1, 4]：第1个样本的第4个位置的表示（768维）
        # x[2, 1]：第2个样本的第1个位置的表示（768维）
        # 最终结果形状为 (3, 768)，即每个样本一个向量表示
        x = x[torch.arange(x.size(0)), reward_idxs] # (b, n_embd)
        rewards = self.prediction_head(x).squeeze(1) # (b,)     使用预测头将transformer的输出映射到标量奖励值

        if positive_tokens is not None:
            s = positive_tokens.size(0)
            rejected = rewards[:s]      # 负样本的奖励值
            chosen = rewards[s:]        # 正样本的奖励值
            # chosen - rejected：正样本奖励减去负样本奖励
            # logsigmoid：对差异应用log sigmoid函数
            # 当正样本奖励高于负样本奖励时，chosen - rejected 为正值，logsigmoid 接近0，损失接近0。
            # 当正样本奖励低于负样本奖励时，chosen - rejected 为负值，logsigmoid 为负值，损失为正值。
            loss = -torch.mean(nn.functional.logsigmoid(chosen - rejected))
            acc = (chosen > rejected).float().detach().mean()   # 计算准确率
            return loss, acc

        return rewards
