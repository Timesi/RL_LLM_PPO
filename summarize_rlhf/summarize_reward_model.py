import os
import time
import sys

# 将项目根目录添加到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import datasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.bpe import BPETokenizer
from mingpt.logger import Logger
from mingpt.model import GPT
from mingpt.rewards import RewardModel
from mingpt.trainer import CN
from mingpt.utils import set_seed, try_auto_cast


class RewardModelSummarize(Dataset):
    def __init__(self, split: str, block_size: int = 1024):
        self.split = split
        self.tokenizer = BPETokenizer()
        self.voc = 50257
        self.block_size = block_size
        ds = datasets.load_dataset("CarperAI/openai_summarize_comparisons", split=split)
        def drop_long_examples(examples):
            prompts = []
            all_chosen = []
            all_rejected = []
            for prompt, chosen, rejected in zip(examples['prompt'], examples['chosen'], examples['rejected']):
                prompt_len = self.tokenizer(prompt).size(1)
                chosen_len = self.tokenizer(chosen).size(1)
                rejected_len = self.tokenizer(rejected).size(1)
                if prompt_len + chosen_len <= block_size + 1 and prompt_len + rejected_len <= block_size + 1:
                    prompts.append(prompt)
                    all_chosen.append(chosen)
                    all_rejected.append(rejected)

            return {"prompt": prompts, "chosen": all_chosen, "rejected": all_rejected}

        self.ds = ds.map(drop_long_examples, batched=True, remove_columns=ds.column_names, num_proc=os.cpu_count())

    def __len__(self):
        return len(self.ds)

    def get_vocab_size(self):
        return self.voc

    def get_block_size(self):
        return self.block_size

    def pad_toks(self, toks):
        # The padding here differs from the SFT since we don't need the LM targets
        mask = torch.full((self.block_size,), False, dtype=bool)
        if len(toks) >= self.block_size:
            toks = toks[-self.block_size:]
        else:
            pad = torch.full((self.block_size,), self.tokenizer.eot_token, dtype=torch.long)
            pad[:len(toks)] = toks

            # include a final eot token to predict
            mask[len(toks) + 1:] = True
            toks = pad

        return toks, mask

    def __getitem__(self, idx):
        row = self.ds[idx]
        prompt, chosen, rejected = row['prompt'], row['chosen'], row['rejected']
        prompt = self.tokenizer(prompt).squeeze(0)
        chosen = self.tokenizer(chosen).squeeze(0)
        rejected = self.tokenizer(rejected).squeeze(0)

        chosen, cmask = self.pad_toks(torch.cat((prompt, chosen)))
        rejected, rmask = self.pad_toks(torch.cat((prompt, rejected)))

        return {
            "pos_toks": chosen,
            "pos_mask": cmask,
            "neg_toks": rejected,
            "neg_mask": rmask,
        }


@torch.no_grad()
def evaluate(model, config, ds, iters=32):
    train_loader = DataLoader(
        ds,
        shuffle=False,
        batch_size=config.batch_size * 2,
        drop_last=True
    )
    total_loss = 0
    total_acc = 0
    i = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, acc = model(
            batch["neg_toks"],
            attn_mask=batch["neg_mask"],
            positive_tokens=batch["pos_toks"],
            positive_mask=batch["pos_mask"]
        )
        total_loss += loss.item()
        total_acc += acc.item()
        i += 1
        if i == iters:
            break
    return total_loss / i, total_acc / i


@torch.no_grad()
def set_reward_bias(model, config, ds, iters=128):
    train_loader = DataLoader(
        ds,
        shuffle=False,
        batch_size=config.batch_size * 2,
        drop_last=True
    )
    all_rewards = []
    i = 0
    for batch in train_loader:
        x = torch.cat((batch["pos_toks"], batch["neg_toks"]))
        mask = torch.cat((batch["pos_mask"], batch["neg_mask"]))
        x, mask = [v.to(device) for v in (x, mask)]
        rewards = model(x, attn_mask=mask)
        all_rewards.append(rewards)
        i += 1
        if i == iters:
            break

    reward_bias = torch.mean(torch.cat(all_rewards))
    model.prediction_head.bias.sub_(reward_bias)
    print("Set reward bias to", model.prediction_head.bias.item())


def run(model, config, logger):
    # setup the dataloader
    train_loader = DataLoader(      # 设置数据加载器
        train_dataset,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model.train()   # 训练模式
    iter_num = 0    # 迭代次数
    iter_time = time.time()     # 记录时间

    for epoch in range(epochs):     # 训练轮数
        for i, batch in enumerate(train_loader):        # 数据加载
            batch = {k: v.to(device) for k, v in batch.items()}     # 移动数据到设备

            # forward the model
            with try_auto_cast(device):     # 尝试自动转换数据类型
                loss, acc = model(
                    batch["neg_toks"],
                    attn_mask=batch["neg_mask"],
                    positive_tokens=batch["pos_toks"],
                    positive_mask=batch["pos_mask"]
                )

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)   # 清空梯度
            loss.backward()                     # 反向传播
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)       # 梯度裁剪
            optimizer.step()                    # 更新参数

            # Evaluation
            if i % 30 == 0:
                model.eval()
                tnow = time.time()
                iter_dt = tnow - iter_time
                val_loss, val_acc = evaluate(model, config, valid_dataset)
                print(f"E: {epoch}, Iter: {iter_num}, Train loss: {loss.item():.4f}, Train Acc: {acc.item():.2f}, Grad norm: {grad_norm:.4f}, Val loss: {val_loss:.4f}, Val Acc: {val_acc:.2f} Took: {iter_dt:.0f}s")
                iter_time = tnow
                model.train()

                # Collect data for plotting
                logger.log("Val Loss", iter_num, val_loss)
                logger.log("Val Acc", iter_num, val_acc)

            logger.log("Train Loss", iter_num, loss.item())
            logger.log("Train Acc", iter_num, acc.item())

            iter_num += 1

    set_reward_bias(model, config, train_dataset)


if __name__ == "__main__":
    set_seed(424242)    # 设置随机种子
    torch.set_float32_matmul_precision('high')      # 控制浮点数矩阵乘法的精度

    # simple supervised training loop for the reward model!

    # ------
    # Config
    # ------
    config = CN()

    # 训练参数
    config.num_workers = 4
    config.batch_size = 48
    config.learning_rate = 3e-6
    config.betas = (0.9, 0.95)
    config.weight_decay = 0.1
    config.grad_norm_clip = 1.0

    epochs = 1
    block_size = 1024
    train_dataset = RewardModelSummarize(block_size=block_size, split='train')
    valid_dataset = RewardModelSummarize(block_size=block_size, split='valid1')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device", device)

    print("train dataset:", len(train_dataset), "val dataset:", len(valid_dataset))

    # 模型参数
    config.model = GPT.get_default_config()     # 获取GPT默认配置
    config.model.model_type = "gpt2"
    config.model.n_layer = 12
    config.model.n_head = 12
    config.model.n_embd = 768
    config.model.resid_pdrop = 0
    config.model.attn_pdrop = 0
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = RewardModel(config.model)       # 根据配置创建RewardModel模型

    # Load model from finetuned model
    sd = torch.load("summarize_sft.pt")
    missing = model.load_state_dict(sd, strict=False)
    # We expect the lm head to be replaced by the scalar reward prediction head
    print("Missing keys:", missing)

    # setup for logging
    logger = Logger()

    model.to(device)    # 将模型移动到对应设备
    model.train()   # 训练模式
    uncompiled_model = model    # 保存原始模型
    model = torch.compile(model)    # 使用torch.compile进行加速

    # setup the optimizer
    optimizer = GPT.configure_optimizers(model, config)     # 配置优化器

    # -----
    # Train
    # -----
    # to speed up training, we can start by only training the reward head of the model
    # model.transformer.requires_grad_(False)
    # run(model, config, logger)

    config.batch_size = 8       # 设备batch_size
    model.requires_grad_(True)  # 将模型参数设置为可训练
    run(model, config, logger)  # 运行训练

    torch.save(uncompiled_model.state_dict(), "reward_model.pt")    # 保存模型
    logger.plot({"Loss": ["Train Loss", "Val Loss"], "Accuracy": ["Train Acc", "Val Acc"]}, filename="reward_model.png")
