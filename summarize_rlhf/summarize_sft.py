import os
import sys

# 将项目根目录添加到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import datasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

from summarize_gpt import SummarizePrompt


class SFTSummarize(Dataset):
    def __init__(self, split: str, block_size: int = 1024):
        # 划分数据集
        self.split = split
        # 使用BPE分词器，将文本转换为token
        self.tokenizer = BPETokenizer()
        # 词汇表大小
        self.voc = 50257
        # 序列最大长度
        self.block_size = block_size
        # 加载“文章-摘要”数据集
        ds = datasets.load_dataset("CarperAI/openai_summarize_tldr", split=split)
        # 过滤掉“提示词+输出”大于输出序列长度的样本
        def drop_long_examples(examples):
            prompts = []    # 存储过滤后的文本提示词
            completions = []    # 存储对应的完摘要文本
            for prompt, completion in zip(examples['prompt'], examples['label']):
                if self.tokenizer(prompt + completion).size(1) <= block_size + 1:
                    prompts.append(prompt)
                    completions.append(completion)

            return {"prompt": prompts, "completion": completions}
        # 使用map函数批量处理数据
        self.ds = ds.map(drop_long_examples, batched=True, remove_columns=ds.column_names, num_proc=os.cpu_count())

    def __len__(self):
        return len(self.ds)

    def get_vocab_size(self):
        return self.voc

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # 分别对提示词和完成项进行编码
        prompt = self.tokenizer(sample["prompt"]).squeeze(0)
        completion = self.tokenizer(sample["completion"]).squeeze(0)
        # 将提示词和完成项拼接成完整的序列
        toks = torch.cat((prompt, completion))

        # attend to all tokens except the padding tokens
        # 初始化一个长度为1025的一维张量（注意力掩码），数据类型为布尔值，填充值为False
        mask = torch.full((self.block_size + 1,), False, dtype=bool)

        # 如果编码后的token数量大于或等于block_size + 1，则截取最后block_size + 1个token
        if len(toks) >= self.block_size + 1:
            toks = toks[-self.block_size - 1:]
        else:
            # 否则，创建一个填充token的张量，长度为block_size + 1，填充值为tokenizer的eot_token（序列结束token）
            pad = torch.full((self.block_size + 1,), self.tokenizer.eot_token, dtype=torch.long)
            pad[:len(toks)] = toks      # 将原始token复制到填充张量的前面

            # include a final eot token to predict
            mask[len(toks) + 1:] = True     # 在掩码中标记填充token的位置为True
            toks = pad                      # 更新toks为填充后的张量

        x = toks[:-1]           # 准备模型的输入x，即除了最后一个token之外的所有token
        y = toks[1:].clone()    # 准备模型的目标y，即除了第一个token之外的所有token，并复制一份以避免修改原始数据

        # we only use the completion tokens to learn on
        # 在目标中，将填充token所在设置为-1，这样在计算损失时会忽略这些填充token
        y[mask[1:]] = -1 # ignore the loss from padding tokens
        # y[:len(prompt)-1] = -1 # and ignore the loss from the prompt tokens
        return x, y, mask[:-1]

def batch_end_callback(trainer):
    model = trainer.model
    model.eval()    # 设置为评估模式
    # 记录训练损失
    trainer.logger.log("Train", trainer.iter_num, trainer.loss.item())

    # 定期验证模型性能
    if trainer.iter_num % trainer.config.log_every == 0:
        # evaluate both the train and test score
        with torch.no_grad():
            total_loss = 0
            for i, batch in enumerate(valid_loader):
                batch = [x.to(device) for x in batch]
                logits, loss = model(*batch)    # 计算验证损失
                total_loss += loss.item()

        val_loss = total_loss / (i+1)
        trainer.logger.log("Valid", trainer.iter_num, val_loss)
        print(f"E: {trainer.epoch}, iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}, val loss: {val_loss:.5f}")

    if trainer.iter_num % trainer.config.generate_every == 0:
        with torch.no_grad():
            sample_prompt = prompt_ds[0].to(device)
            idx = model.generate(sample_prompt, max_new_tokens=128, do_sample=True, temperature=0.7).cpu()
            for j,generation in enumerate(idx):
                print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    # save the latest model
    if trainer.config.save_every and trainer.iter_num % trainer.config.save_every == 0:
        print("saving model")
        ckpt_path = os.path.join(os.path.curdir, "model.pt")
        torch.save(model.state_dict(), ckpt_path)

    # 恢复模型为训练模式
    model.train()


if __name__ == '__main__':
    set_seed(424242)        # 设置随机种子
    torch.set_float32_matmul_precision('high')      # 控制浮点数矩阵乘法的精度

    print("===== STARTING PRETRAINING =====")

    # For Logging
    # 记录训练和验证过程中损失值的空列表
    train_idx = []
    train_losses = []
    val_idx = []
    val_losses = []

    valid_iters = 32    # 设置验证过程中使用的迭代次数
    block_size = 1024   # 设置模型的上下文长度
    train_ds = SFTSummarize(block_size=block_size, split='train')       # 加载训练数据集
    valid_ds = SFTSummarize(block_size=block_size, split='valid')       # 加载验证数据集
    prompt_ds = SummarizePrompt(block_size=block_size, split='valid')   # 加载提示数据集

    model_config = GPT.get_default_config()     # 获取GPT默认配置
    model_config.model_type = 'gpt2'
    model_config.vocab_size = train_ds.get_vocab_size()
    model_config.block_size = block_size
    model = GPT.from_pretrained("gpt2")     # 根据参数创建GPT模型

    train_config = Trainer.get_default_config()     # 获取训练器的默认配置对象
    train_config.learning_rate = 5e-6
    train_config.num_workers = 4
    train_config.log_every = 500
    train_config.generate_every = 1000
    train_config.save_every = None
    train_config.epochs = 3
    train_config.batch_size = 8
    train_config.compile = True
    trainer = Trainer(train_config, model, train_ds)    # 设置训练器

    device = trainer.device     # 获取训练设备
    valid_loader = DataLoader(  # 创建用于验证的DataLoader
        valid_ds,
        shuffle=False,
        num_workers=2,
        batch_size=trainer.config.batch_size * 2,       # 验证批次大小是训练批次大小的2倍，因为验证时不需要计算梯度，可以使用更大的批次
    )

    trainer.set_callback('on_batch_end', batch_end_callback)    # 设置回调函数
    trainer.run()   # 启动训练

    print("===== DONE PRETRAINING =====")

    # Get a validation prompt to test
    sample_prompt = prompt_ds[0].to(device)     # 从提示数据集中获取第一个样本作为输入提示
    idx = model.generate(sample_prompt, max_new_tokens=128, do_sample=True, top_k=30, stop_at=train_ds.tokenizer.eot_token).cpu()   # 模型生成
    for j,generation in enumerate(idx):
        print(f"Generation {j}:", train_ds.tokenizer.decode(generation))

    # Plot the losses
    trainer.logger.plot({"Loss": ["Train", "Valid"]}, filename="summarize_sft.png")

    # Save the Model
    torch.save(model.state_dict(), "summarize_sft.pt")
