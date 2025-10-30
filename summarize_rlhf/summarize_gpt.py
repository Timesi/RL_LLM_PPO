# The summarization task differs from the previous happyGPT in a few small ways:
# 1. use the summarization dataset. pretty straight forward containing 100000 summarizations.
#    each datapoint has a prompt and a summary completion. for sft we train only the completion.
#    we also use the gpt-2 tokenizer. also uses mask.
# 2. a proper reward model, not just a heuristic sentiment. the reward model is initialized from
#    pretrained LM and contains a new 'head' layer that predicts the reward. we only train this
#    linear layer of the reward model.
# 3. full ppo optimization to match the instructgpt paper. ppo is very similar to our initial
#    implementation but it prevents the model from updating too far away from the prior distribution
#    using a clipped loss. also uses advantage estimation and a separate value function

import copy
import os
import sys

# 将项目根目录添加到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import datasets
import torch
from torch.utils.data import Dataset
from tqdm import trange

from mingpt.bpe import BPETokenizer
from mingpt.logger import Logger
from mingpt.model import GPT
from mingpt.rewards import (RewardModel, ValueModel,
                            calculate_advantage_and_returns)
from mingpt.utils import lr_schedule, masked_mean, set_seed, try_auto_cast


class SummarizePrompt(Dataset):
    def __init__(self, split: str, block_size: int = 1024):
        self.split = split
        self.tokenizer = BPETokenizer()     # 初始化BPE分词器
        self.voc = 50257                    # 词汇表大小
        self.block_size = block_size        # 最大长度限制
        ds = datasets.load_dataset("CarperAI/openai_summarize_tldr", split=split)   # 从huggingface加载数据集
        # 过滤掉超出长度范围的数据
        def drop_long_examples(examples):
            prompts = []
            for prompt in examples['prompt']:
                if self.tokenizer(prompt).size(1) <= block_size:
                    prompts.append(prompt)

            return {"prompt": prompts}

        # 批量处理数据
        self.ds = ds.map(
            drop_long_examples,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=os.cpu_count()
        )
        if len(self.ds) > 50000:
            self.ds = ds.select(range(50000))


    def __len__(self):
        return len(self.ds)

    def get_vocab_size(self):
        return self.voc

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        sample = self.ds[idx]
        return self.tokenizer(sample["prompt"])


def validate(valid_ds, model, reward_model, device, max_iters=64):
    reward_model.eval()
    model.eval()

    total_rewards = 0
    count = 0
    total = min(max_iters, len(valid_ds))
    valid_progress_bar = trange(total, desc="Validating", leave=False)
    with torch.no_grad():
        for i in valid_progress_bar:
            prompt = valid_ds[i].to(device)

            # 从验证数据集中获取第[i]个样本（提示词），并将其移动到指定的计算设备上，生成一个完成项
            completion = model.generate(prompt, max_new_tokens=completion_len, do_sample=True, top_k=30, stop_at=end_of_text)

            # 使用奖励模型评估生成的完成项，并将结果转换为Python标量值
            reward = reward_model(completion).item()
            # 将当前奖励值累加到总奖励中
            total_rewards += reward
            count += 1
            valid_progress_bar.set_postfix(avg_reward=f"{total_rewards/count:.4f}")

            if i < 3:
                print(train_ds.tokenizer.decode(completion[0]), f"\nReward: {reward}\n========\n")

     # 计算验证样本的平均奖励值
    average_reward = total_rewards / total
    model.train()
    return average_reward


def ppo(model, reward_model, value_model, ref_model, logger):
    # PPO training loop:
    # 1. generate a set of completions, given some prompts for the task
    # 2. calculate the rewards, values and advantages of the completions
    # 3. optimize the models completions based on the rewards using ppo objective

    # 在训练开始前，使用验证函数评估初始模型的性能，并打印初始验证奖励值
    val_reward = validate(valid_ds, model, reward_model, device)
    print("Initial (SFT) Val reward:", val_reward)

    i = 0
    for epoch in range(n_epochs):
        batch_idxs = torch.randperm(len(train_ds))      # 生成随机排列的训练数据索引，用于打乱训练数据顺序
        for idx in trange(len(train_ds) // sample_batch_size, desc="iter"):     # 遍历所有批次，每个批次包含sample_batch_size个样本

            # Learning rate schedule
            # 根据当前迭代次数更新优化器中的学习率
            curr_lr = get_lr(i)
            for pg in optimizer.param_groups:
                pg['lr'] = curr_lr

            # Sample completions given some prompts from the dataset
            # these are the `actions` in the RL sense that the model takes
            with torch.no_grad():   # 关闭梯度计算，开始采样阶段，这些生成的完成项在强化学习中被视为“action”
                model.eval()
                value_model.eval()

                original_log_probs = []     # 原始对数概率
                completions = []            # 生成的完成项
                advantages = []             # 优势值
                returns = []                # 回报值
                action_mask = []            # 动作掩码
                targets = []                # 目标序列
                total_reward = 0            # 总奖励
                start_idx = idx * sample_batch_size     # 计算当前批次的起始索引
                for prompt_idx in batch_idxs[start_idx : start_idx + sample_batch_size]:        # 遍历该批次中的所有样本索引
                    prompt = train_ds[prompt_idx.item()].to(device)

                    # Sample the completions
                    completion = model.generate(prompt, max_new_tokens=completion_len, do_sample=True, top_k=30, stop_at=end_of_text)   # 使用策略模型生成完成项

                    # 检查生成的完成项是否以结束标记结束
                    if completion[0, -1] == end_of_text:
                        # Evaluate and store the rewards for the last token
                        reward = reward_model(completion).unsqueeze(-1)     # 如果是，则使用奖励模型评估完成项

                    else:
                        # If there is no eot token, hardcode a negative reward
                        reward = torch.tensor([[-1.0]], device=device)      # 如果不是，则给予-1的负奖励（惩罚）

                    total_reward += reward.item()       # 累加当前样本的奖励值

                    completion_minus_1, target = completion[:, :-1], completion[:, 1:]      # 准备计算对数概率所需要的的输入和目标序列

                    # Store the model's original log prob (could be merged into the generate fn)
                    original_log_prob = model.log_probs(completion_minus_1, target)         # 使用策略模型计算并存储当前策略模型对生成序列的对数概率

                    # Reference logprobs
                    ref_log_prob = ref_model.log_probs(completion_minus_1, target)          # 使用参考模型计算对生成序列的对数概率，用于后续计算KL散度

                    # Calculate values, returns and advantages
                    values = value_model(completion)                                        # 使用价值模型计算当前完成项的价值

                    # Calculate the advantage for our policy gradient
                    # Include the kl score to reduce overfitting
                    # the kl reward here could be kept up to date with the policy network
                    # inside the ppo updates below for a better regularization effect
                    # 计算优势函数
                    # 1、计算KL散度
                    # 2、构造包含KL惩罚和奖励的得分向量
                    # 3、使用GAE算法计算优势值和回报值
                    kl = original_log_prob - ref_log_prob
                    score = torch.cat((- kl_beta * kl, reward), dim=1)
                    advantage, single_return = calculate_advantage_and_returns(score, values, gamma=gamma, lambd=lambd)

                    # Pad the values up to block_size with zeros
                    # 对优势值和回报值进行填充，使其长度达到block_size
                    pad = torch.zeros(1, block_size - advantage.size(1), device=advantage.device, dtype=advantage.dtype)
                    advantages.append(torch.cat((advantage, pad), dim=1))
                    returns.append(torch.cat((single_return, pad), dim=1))

                    # pad the log probs with 1 extra 0
                    # 对原始对数概率进行填充
                    pad_plus_1 = torch.zeros(1, block_size - original_log_prob.size(1), device=advantage.device, dtype=advantage.dtype)
                    original_log_probs.append(torch.cat((original_log_prob, pad_plus_1), dim=1))

                    # Pad the tokens with longs
                    # 对生成的token序列和目标序列进行填充
                    pad = torch.zeros(1, block_size - completion.size(1), device=completion.device, dtype=completion.dtype)
                    completions.append(torch.cat((completion, pad), dim=1))
                    pad = torch.zeros(1, block_size - target.size(1), device=target.device, dtype=target.dtype)
                    targets.append(torch.cat((target, pad), dim=1))

                    # The action mask is only the generated part of the completion
                    # 创建动作掩码，只标记生成部分（而非提示部分）为1
                    mask = torch.zeros(1, block_size, device=advantage.device, dtype=advantage.dtype)
                    mask[:, prompt.size(1):completion.size(1)] = 1
                    action_mask.append(mask)

            # Stack the values into a batch
            # 将收集的所有数据按批次堆叠成张量
            advantages = torch.cat(advantages)
            returns = torch.cat(returns)
            completions = torch.cat(completions)
            original_log_probs = torch.cat(original_log_probs)
            action_mask = torch.cat(action_mask)
            targets = torch.cat(targets)


            # Do the PPO update on the batch of data several times
            # 将模型设置为训练模式，准备进行参数更新
            model.train()
            value_model.train()
            for _ in range(n_updates):      # 每次采样后更新n_updates次参数
                b_inds = torch.randperm(sample_batch_size)
                for start in range(0, sample_batch_size, train_batch_size):
                    end = start + train_batch_size

                    # Grab the mini-batches
                    # 进行多次PPO更新，每次更新都遍历所有小批次数据
                    mb_inds = b_inds[start:end]
                    mb_completion = completions[mb_inds]
                    mb_target = targets[mb_inds]
                    mb_original_logps = original_log_probs[mb_inds]
                    mb_advantages = advantages[mb_inds]
                    mb_returns = returns[mb_inds]
                    mb_action_mask = action_mask[mb_inds]

                    # TODO: Masked normalize advantages
                    # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    with try_auto_cast(device):
                        # Forward pass through the latest model
                        log_probs = model.log_probs(mb_completion, mb_target)

                        # Policy loss
                        # 计算PPO策略损失：
                        # 1、计算对数概率比率
                        # 2、计算两种形式的策略梯度损失（PPO的裁剪机制）
                        # 3、取两者的最大值作为最终策略损失
                        logratio = log_probs - mb_original_logps
                        ppo_ratio = logratio.exp()
                        pg_loss1 = -mb_advantages * ppo_ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ppo_ratio, 0.8, 1.2)
                        pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mb_action_mask, dim=1).mean()

                        # Value loss
                        # 使用均方误差计算价值损失
                        new_value = value_model(mb_completion)
                        v_loss = 0.5 * masked_mean((new_value - mb_returns) ** 2, mb_action_mask, dim=1).mean()

                        # 合并策略损失和价值损失，并根据梯度累积步数进行缩放
                        loss = pg_loss + 0.1 * v_loss
                        loss = loss / grad_accum_steps

                    loss.backward()

                # 进行梯度裁剪、优化器步进和梯度清零操作
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                value_grad_norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), grad_norm_clip)
                optimizer.step()
                model.zero_grad()
                value_model.zero_grad()


            clipfrac = ((ppo_ratio - 1.0).abs() > 0.2).float().mean().item()
            avg_reward = total_reward / sample_batch_size
            logger.log("Clip Frac", i, clipfrac)
            logger.log("Reward", i, avg_reward)
            logger.log("Value Loss", i, v_loss.item())
            logger.log("KL", i, kl.mean().item())
            logger.log("Policy Grad Norm", i, policy_grad_norm.item())
            logger.log("Value Grad Norm", i, value_grad_norm.item())

            if i % 20 == 0:
                val_reward = validate(valid_ds, model, reward_model, device)
                logger.log("Val Reward", i, val_reward)
                print(f"Iter: {i}, Avg reward: {avg_reward:.3f}, KL: {kl.mean().item():.3f}, Value Loss: {v_loss.item():.4f}, Grad Norm: {policy_grad_norm:.2f}, Vf grad norm: {value_grad_norm:.2f}, Val reward: {val_reward:.3f}")

                # intermediate plots help with debugging!
                logger.plot({"Reward": ["Reward", "Val Reward"], "Value Loss": ["Value Loss"]}, filename="summarize_rl_rewards.png")
                logger.plot({"Gradient Norm": ["Policy Grad Norm", "Value Grad Norm"], "KL Div": ["KL"]}, filename="summarize_rl_metrics.png")

                torch.save(uncompiled_model.state_dict(), "summarize_rl.pt")

            i += 1


if __name__ == "__main__":
    set_seed(424242)
    torch.set_float32_matmul_precision('high')

    # ------
    # Config
    # ------

    # Transformer context len
    # For GPT-2, should be 1024
    block_size = 1024

    # completion + prompt <= block_size
    completion_len = 80
    max_prompt_len = 512

    model = GPT.get_default_config()
    model.model_type = "gpt2"
    model.n_layer = 12
    model.n_head = 12
    model.n_embd = 768
    model.vocab_size = 50257
    model.model_type = None
    model.block_size = block_size

    reward_model = RewardModel(model)   # 配置奖励模型
    value_model = ValueModel(model)     # 配置价值模型
    model = GPT(model)                  # 配置策略模型

    # Load reward, value, and model from weights!
    model.load_state_dict(torch.load("summarize_sft.pt", map_location='cpu'))
    reward_model.load_state_dict(torch.load("reward_model.pt", map_location='cpu'))
    value_model.load_state_dict(torch.load("reward_model.pt", map_location='cpu'))

    # reference model is the finetuned SFT model
    ref_model = copy.deepcopy(model)    # 深拷贝策略模型用于计算KL散度惩罚项，确保新策略不会偏离原始策略太远
    ref_model.requires_grad_(False)     # 冻结策略模型参数
    ref_model.eval()                    # 设置为评估模式

    reward_model.requires_grad_(False)  # 冻结奖励模型参数，奖励模型是预先训练好的，用于评估生成文本的质量，不需要训练
    reward_model.eval()                 # 设置为评估模式

    # PPO hyperparams
    # 在RLHF训练中，我们需要从当前策略模型中采样一批输出来计算奖励和执行策略更新
    # 这里的64表示每次训练迭代中，我们会让模型生成64个完整的回应（例如摘要）
    sample_batch_size = 64 # Number of completions to sample
    # 采样了64个完成项，拆分为多个训练批次，训练批次大小为8
    train_batch_size = 8 # OAI uses equal training and sampling batches of 64 (we'll use whatever fits on the GPU!)
    # 由于采样批次(64)大于训练批次(8)，需要进行梯度累积以模拟更大的批次效果。
    # 64除以8等于8，意味着需要8次前向和后向传播才会执行一次优化器步骤。这样做可以在有限的GPU内存下模拟大批量训练的效果
    grad_accum_steps = sample_batch_size // train_batch_size
    # 学习率，选择相对较小的学习率，适用于微调大型语言模型，避免破坏预训练模型已经学到的知识。
    max_learning_rate = 3e-6
    # 设置梯度裁剪，当梯度的范数超过这个阈值时，会将其按比例缩小，保证训练的稳定性，防止梯度爆炸
    grad_norm_clip = 1.0
    # 设置KL散度惩罚项的系数为0.02
    # 在RLHF中，为了防止新策略偏离原始策略太远，会在目标函数中加入KL散度惩罚项
    # 这个系数控制着KL散度项在总损失中的权重，值越大表示对策略变化的约束越强
    kl_beta = 0.02
    # 每次采样后策略更新的次数，也就是说，对于每批采样的数据，进行2轮参数更新
    n_updates = 2
    # 设置折扣因子，这个系数用于计算未来奖励的权重，值越大表示未来奖励越重要
    gamma = 1
    # 设置GAE的lambd参数为0.95。GAE是一种用于计算优势函数的方法，它在偏差和方差之间进行权衡。
    lambd = 0.95
    # 设置训练周期
    n_epochs = 1

    # Logging
    logger = Logger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    value_model.to(device)
    reward_model.to(device)
    ref_model.to(device)
    print("Running on device", device)

    uncompiled_model = model
    uncompiled_value_model = value_model

    # compile the model
    model = torch.compile(model)
    reward_model = torch.compile(reward_model)
    value_model = torch.compile(value_model)
    ref_model = torch.compile(ref_model)

    # Prompt datasets
    train_ds = SummarizePrompt('train', block_size=max_prompt_len)
    valid_ds = SummarizePrompt('valid', block_size=max_prompt_len)
    end_of_text = train_ds.tokenizer.eot_token
    print("train ds:", len(train_ds), "val ds:", len(valid_ds))

    # Can set separate lrs for policy and value fn
    # 使用训练数据集的总大小/每次采样的批量大小*训练周期=总迭代次数
    total_iters = len(train_ds) // sample_batch_size * n_epochs
    # 创建学习率调度器，用于根据迭代次数调整学习率
    get_lr = lr_schedule(max_learning_rate, max_iters=total_iters)
    optim_groups = [{'params': model.parameters()}, {'params': value_model.parameters()}]
    # 使用AdamW优化器优化参数
    optimizer = torch.optim.AdamW(optim_groups, lr=get_lr(0), betas=(0.9, 0.95), fused=torch.cuda.is_available(), weight_decay=0.0)

    # --------
    # Run PPO!
    # --------
    # I'm just using the global config vars here :)
    ppo(model, reward_model, value_model, ref_model, logger)

    torch.save(uncompiled_model.state_dict(), "summarize_rl.pt")
    torch.save(uncompiled_value_model.state_dict(), "value_model.pt")

    # Plot the results
    logger.plot({"Reward": ["Reward", "Val Reward"], "Value Loss": ["Value Loss"]}, filename="summarize_rl_rewards.png")
    logger.plot({"Gradient Norm": ["Policy Grad Norm", "Value Grad Norm"], "KL Div": ["KL"], "Clip Fraction": ["Clip Frac"]}, filename="summarize_rl_metrics.png")
