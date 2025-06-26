"""
训练和评估函数，用于在main.py中调用
"""
import json
import math
import os
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from losses import DistillationLoss
import utils
from sklearn.metrics import f1_score, roc_auc_score  # 导入 f1_score
from sklearn.metrics import recall_score
import numpy as np

def set_bn_state(model):
    """
    设置批归一化层的状态，使其处于评估模式。
    """
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_one_epoch(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    clip_grad: float = 0,
    clip_mode: str = 'norm',
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    set_bn_eval=False,
):
    """
    训练模型一个周期，更新模型参数并记录训练过程中的各项指标。

    参数:
    - model: 要训练的神经网络模型
    - criterion: 损失函数，使用知识蒸馏损失
    - data_loader: 数据加载器，提供训练数据
    - optimizer: 优化器，用于更新模型参数
    - device: 设备（CPU或GPU）
    - epoch: 当前周期编号
    - loss_scaler: 损失缩放器，用于混合精度训练
    - clip_grad: 梯度裁剪的阈值
    - clip_mode: 梯度裁剪的模式
    - model_ema: 模型的指数移动平均
    - mixup_fn: 混合增强方法
    - set_training_mode: 设置模型是否处于训练模式
    - set_bn_eval: 是否设置批归一化层为评估模式
    """

    # 设置模型为训练模式（根据参数设定）
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)

    # 创建指标记录器，并添加学习率指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 100

    # 遍历数据集
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # 将数据移至设备
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 如果使用 mixup 数据增强，则应用混合操作
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # 前向传播，计算模型输出和损失
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        # 获取损失值
        loss_value = loss.item()

        # 如果损失值无效，则停止训练
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # 清零梯度
        optimizer.zero_grad()

        # 判断优化器是否为二阶优化器
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        # 同步 CUDA 操作
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # 更新并记录损失和学习率
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device , epoch, output_dir):
    """
    评估模型性能，计算准确率、损失和F1得分。

    参数:
    - data_loader: 数据加载器，提供评估数据
    - model: 要评估的神经网络模型
    - device: 设备（CPU或GPU）

    返回:
    - 包含平均准确率、F1得分和损失的字典
    """

    # 定义交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 创建指标记录器
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # 切换到评估模式
    model.eval()

    # 保存所有预测和实际标签，用于F1计算
    all_targets = []
    all_preds = []
    all_scores = []

    # 遍历评估数据集
    for images, target in metric_logger.log_every(data_loader, 10, header):
        # 将数据移至设备
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 计算输出和损失
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # 计算 top-1 和 top-5 准确率
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # 获取预测结果并记录用于F1计算
        _, preds = output.topk(1, 1, True, True)
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(preds.squeeze().cpu().numpy())

        # 保存模型的softmax概率输出用于AUC计算
        if output.shape[1] == 2:  # 假设输出有两个类别
            probs = torch.nn.functional.softmax(output, dim=1)
            all_scores.extend(probs[:, 1].cpu().numpy())
        else:
            raise ValueError("AUC calculation assumes binary classification with two output scores.")

        # 更新和记录损失与准确率
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # 计算F1得分
    f1 = f1_score(all_targets, all_preds, average='weighted')
    metric_logger.meters['f1'].update(f1)

    # 计算AUC得分
    auc = roc_auc_score(all_targets, all_scores)
    metric_logger.meters['auc'].update(auc)
    # 计算G-Mean得分
    recall_pos = recall_score(all_targets, all_preds, pos_label=1)
    recall_neg = recall_score(all_targets, all_preds, pos_label=0)
    gmean = np.sqrt(recall_pos * recall_neg)
    metric_logger.meters['gmean'].update(gmean)

    print(
        '* Acc@1 {top1.global_avg:.5f} F1 {f1:.3f} Acc@5 {top5.global_avg:.5f} AUC {auc:.5f} G-Mean {gmean:.5f} loss {losses.global_avg:.5f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, f1=f1, auc=auc, gmean=gmean,
                losses=metric_logger.loss))

    # 组织评估结果
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results_with_epoch = {"epoch": epoch, **results}

    # ---------- 保存 JSON ----------
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f_json:
                history = json.load(f_json)
            if not isinstance(history, list):
                print("⚠️ 检测到旧 JSON 格式不正确，已重置为列表。")
                history = []
        except json.JSONDecodeError:
            print("⚠️ JSON 解码失败，文件可能损坏，已重置为空列表。")
            history = []
    else:
        history = []

    history.append(results_with_epoch)
    with open(json_path, "w") as f_json:
        json.dump(history, f_json, indent=4)

    # ---------- 保存 TXT ----------
    txt_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(txt_path, "a") as f_txt:
        f_txt.write(f"Epoch {epoch}:\n")
        for k, v in results.items():
            f_txt.write(f"{k}: {v:.6f}\n")
        f_txt.write("\n")

    return results