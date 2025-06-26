import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """跟踪一系列值并提供在窗口内或整个序列中的平滑值"""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        # 使用一个双端队列来存储最近的值，限制最大长度为 window_size
        self.deque = deque(maxlen=window_size)
        self.total = 0.0  # 总和
        self.count = 0    # 值的数量
        self.fmt = fmt    # 格式化字符串

    def update(self, value, n=1):
        """更新当前值和总计数"""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """返回窗口内值的中位数"""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """返回窗口内值的平均值"""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """返回整个序列的平均值"""
        return self.total / self.count

    @property
    def max(self):
        """返回窗口内的最大值"""
        return max(self.deque)

    @property
    def value(self):
        """返回当前的最新值"""
        return self.deque[-1]

    def __str__(self):
        """返回格式化的字符串表示"""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """用于记录和显示模型训练过程中的各种指标"""

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)  # 字典存储各个指标的 SmoothedValue
        self.delimiter = delimiter                # 指标分隔符

    def update(self, **kwargs):
        """更新各个指标的数值"""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()  # 如果是张量，转换为 Python 数值
            assert isinstance(v, (float, int))
            self.meters[k].update(v)  # 更新对应指标的值

    def __getattr__(self, attr):
        """获取指定指标"""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """返回所有指标的字符串表示"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        """添加一个新的指标"""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """用于在指定频率下打印训练日志"""
        i = 0
        if not header:
            header = ''
        start_time = time.time()  # 记录开始时间
        end = time.time()         # 每次迭代的起始时间
        iter_time = SmoothedValue(fmt='{avg:.4f}')  # 迭代时间的平均值
        data_time = SmoothedValue(fmt='{avg:.4f}')  # 数据加载时间的平均值
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',          # 预计剩余时间
            '{meters}',            # 当前的指标
            'time: {time}',        # 平均迭代时间
            'data: {data}'         # 平均数据加载时间
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')  # 记录最大显存使用
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)  # 更新数据加载时间
            yield obj  # 返回当前数据
            iter_time.update(time.time() - end)  # 更新迭代时间
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算预计剩余时间
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()  # 记录当前时间
        total_time = time.time() - start_time  # 计算总耗时
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def replace_batchnorm(net):
    """递归地将模型中的 BatchNorm 层替换为 Identity 层"""
    for child_name, child in net.named_children():
        # 如果模块具有 'fuse' 方法，先融合再递归替换
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        # 如果是 BatchNorm2d 层，替换为 Identity 层
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            # 继续递归遍历子模块
            replace_batchnorm(child)
