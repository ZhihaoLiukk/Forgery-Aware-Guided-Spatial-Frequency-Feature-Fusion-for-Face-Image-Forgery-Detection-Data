'''
Build samplers for data loading
'''
import torch
import math

class SimpleSampler(torch.utils.data.Sampler):
    """
    用于重复数据增强的采样器，不需要分布式采样。
    """

    def __init__(self, dataset, shuffle=True, repeat_factor=3):
        """
        初始化采样器
        dataset: 传入的数据集
        shuffle: 是否对数据进行随机打乱
        repeat_factor: 每个样本重复的次数
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.repeat_factor = repeat_factor
        self.num_samples = len(dataset) * repeat_factor  # 计算总样本数量

    def __iter__(self):
        """
        返回数据集索引的迭代器
        """
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # 如果需要打乱顺序，则对索引进行随机排序
            indices = torch.randperm(len(self.dataset)).tolist()

        # 重复每个索引 repeat_factor 次
        indices = indices * self.repeat_factor
        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples  # 返回总样本数
