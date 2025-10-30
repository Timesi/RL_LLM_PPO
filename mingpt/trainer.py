"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

from mingpt.logger import Logger
from mingpt.utils import CfgNode as CN
from mingpt.utils import try_auto_cast


class Trainer:

    @staticmethod
    # 静态方法，返回常用的默认配置
    def get_default_config():
        C = CN()
        C.epochs = 1
        # device to train on
        C.device = 'auto'   # 自动选择设备（GPU或CPU）
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)   # Adam优化器的beta参数
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0  # 梯度裁剪阈值
        C.compile = False       # 是否使用torch.compile优化
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config    # 保存配置对象
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)      # 使用defaultdict存储回调函数列表
        self.logger = Logger()

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        if config.compile:
            self.model = torch.compile(self.model)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    # 添加回调函数到指定事件
    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    # 设置指定事件的回调函数（替换现有列表）
    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    # 触发指定事件的所有回调函数
    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        # 创建优化器
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,    # 把数据直接锁在页内存，加速 GPU 传输（CPU无影响）
            drop_last=True,     # 不足一个 batch 的尾巴直接扔掉，保证批次大小恒定
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()   # 训练模式
        self.iter_num = 0   # 全部迭代计数
        self.iter_time = time.time()    # 计算batch耗时
        for epoch in range(config.epochs):
            self.epoch = epoch      # 把当前 epoch 存起来，供回调用
            for batch in train_loader:
                batch = [t.to(self.device) for t in batch]  # 将一批 tensor 列表放到 GPU/CPU

                # forward the model
                with try_auto_cast(self.device):
                    # 把loss挂到self.loss上，后面的回调函数就能随时读取
                    logits, self.loss = model(*batch)

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)   # 比 optimizer.zero_grad() 更快，省一次 memset
                self.loss.backward()    # 反向传播
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)   # 全局梯度裁剪，防爆炸
                self.optimizer.step()   # 更新参数

                self.trigger_callbacks('on_batch_end')      # 触发回调
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time    # 本次 batch 耗时
                self.iter_time = tnow
