import time
import json

import utils
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from tqdm import tqdm  # 导入 tqdm 以添加进度条
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from model.dwt import DWT
from model.swt import SWT

from model.fcf import FCFNet
from losses import DistillationLoss  # 知识蒸馏损失函数
from data.datasets import build_dataset  # 数据集构建工具
from engine import evaluate  # 训练与评估函数
from data.samplers import SimpleSampler  # 自定义的简单采样器
from data.threeaugment import new_data_aug_generator  # 三重数据增强生成器
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")
warnings.filterwarnings("ignore", message="Argument interpolation should be of type InterpolationMode instead of int.")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.GradScaler.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

def get_args_parser():
    # """定义命令行参数解析器"""
    # parser = argparse.ArgumentParser('ViT_DWT training and evaluation script', add_help=False)
    #
    # # 设置训练和模型参数
    # parser.add_argument('--batch-size', default=, type=int)
    # parser.add_argument('--epochs', default=, type=int)
    # parser.add_argument('--input-size', default=, type=int, help='images input size')
    #
    # # 融合网络参数
    # parser.add_argument('--decode_channels', default=, type=int, help='')
    # parser.add_argument('--num_classes', default=, type=int, help='')
    # parser.add_argument('--dropout', default=, type=float, help='')
    # parser.add_argument('--window_size', default=, type=int, help='')
    #
    # # 模型 EMA 参数
    # parser.add_argument('--model-ema', action='store_true') # 启用或禁用模型 EMA
    # parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    # parser.add_argument('--model-ema-decay', type=float, default=, help='') # EMA 的衰减率（接近 1 表示较缓慢的更新），这个会影响训练速度吗？
    #
    # # 优化器参数
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer')
    # parser.add_argument('--opt-eps', default=, type=float, metavar='EPSILON') # 优化器的 epsilon 值（控制数值稳定性）
    # parser.add_argument('--clip-grad', type=float, default=, metavar='NORM') # 梯度裁剪值，用于限制梯度的大小 量化情况不清楚 default=0.02
    # parser.add_argument('--weight-decay', type=float, default=) # 权重衰减，用于 L2 正则化 default=0.025
    # parser.add_argument('--momentum', type=float, default=, help='Momentum for optimizer') # 动量，用于优化算法（例如 SGD）
    #
    # # 学习率调度参数
    # parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler')    # 学习率调度器类型
    # parser.add_argument('--lr', type=float, default=, metavar='LR')
    # parser.add_argument('--warmup-lr', type=float, default=, metavar='LR') # 分别为热身学习率和最小学习率default=1e-6,
    # parser.add_argument('--min-lr', type=float, default=, metavar='LR')
    # parser.add_argument('--warmup-epochs', type=int, default=, metavar='N') # 热身和冷却的周期数
    # parser.add_argument('--cooldown-epochs', type=int, default=, metavar='N')
    #
    # # 数据增强与混合参数
    # parser.add_argument('--ThreeAugment', default=False, action='store_true') # 是否使用三重数据增强
    # parser.add_argument('--color-jitter', type=float, default=, metavar='PCT') # 颜色抖动幅度
    # parser.add_argument('--aa', type=str, default='', metavar='NAME') # 自动数据增强类型
    # parser.add_argument('--smoothing', type=float, default=, help='Label smoothing (default: 0.1)') # 标签平滑值，用于防止过拟合
    # parser.add_argument('--repeated-aug', action='store_true') # 是否重复增强
    # parser.add_argument('--reprob', type=float, default=, metavar='PCT') # 随机擦除概率
    # parser.add_argument('--mixup', type=float, default=) # 用于混合增强的系数、概率、模式
    # parser.add_argument('--mixup_prob', type=float, default=, help='Probability for mixup')
    # parser.add_argument('--mixup_mode', type=str, default='batch', choices=['batch', 'pair'], help='Mode for mixup')
    #
    # # 知识蒸馏参数
    # parser.add_argument('--teacher-model', default='teacher-model', type=str, metavar='MODEL') # 教师模型的类型和路径，用于知识蒸馏
    # parser.add_argument('--teacher-path', type=str,default='teacher-model path')
    # parser.add_argument('--distillation_type', default='', choices=['none', 'soft', 'hard'], type=str) # 蒸馏类型（none、soft、hard），决定如何蒸馏 原来参数hard
    # parser.add_argument('--distillation-alpha', default=, type=float) # 蒸馏损失权重，控制基础损失和蒸馏损失的比重
    # parser.add_argument('--distillation-tau', default=, type=float) # 蒸馏温度，用于控制软标签的平滑程度
    #
    # # 微调参数
    # parser.add_argument('--finetune', default='',help='finetune from checkpoint')
    # parser.add_argument('--set_bn_eval', action='store_true', default=False,help='set BN layers to eval mode during finetuning.')
    #
    # # 数据集和路径参数
    # parser.add_argument('--data-path', default='your datasets path', type=str) # 数据集路径和名称
    # parser.add_argument('--data-set', default='your dataset',  type=str)
    # parser.add_argument('--output_dir', default='') # 输出目录，用于保存模型检查点和日志
    # parser.add_argument('--device', default='cuda:') # 设备类型
    # parser.add_argument('--seed', default=0, type=int) # 随机种子，确保结果可复现
    # parser.add_argument('--resume', default='', help='resume from checkpoint') # 从检查点恢复训练
    # parser.add_argument('--eval', action='store_true', help='Perform evaluation only') # 只执行评估而不进行训练
    # parser.add_argument('--num_workers', default=, type=int) # 加载数据的线程数
    # parser.add_argument('--pin-mem', action='store_true') # 是否将数据加载到固定内存中
    #
    # parser.add_argument('--train_interpolation', type=str, default='bilinear', help='training interpolation method')
    # parser.add_argument('--remode', type=str, default='pixel', help='re-mode setting')
    # parser.add_argument('--recount', type=int, default=0, help='re-count setting')
    #
    # # 保存和日志参数
    # parser.add_argument('--save_freq', default=, type=int) # 保存频率
    # parser.add_argument('--project', default='FAM_SFF', type=str)# 项目名称
    # # 在 get_args_parser() 中添加 clip_mode 参数
    # parser.add_argument('--clip-mode', type=str, default='norm', help='Gradient clipping mode')

    return parser

def main(args):

    device = torch.device(args.device)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # 数据集构建
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=SimpleSampler(dataset_train, shuffle=True, repeat_factor=3),  # 使用自定义采样器
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    # 输出数据集加载情况
    print(f"Training dataset loaded with {len(dataset_train)} samples.")
    print(f"Validation dataset loaded with {len(dataset_val)} samples.")

    # 配置 Mixup 数据增强
    mixup_fn = None
    mixup_active = args.mixup > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, prob=args.mixup_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 模型
    print(f"Instantiating ParallelFaceForgeryDetection model")
    model = FCFNet(decode_channels=args.decode_channels, num_classes=args.num_classes,
                dropout=args.dropout, window_size=args.window_size, distillation=args.distillation_type)

    # finetune，加载预训练权重
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            print(f"Loading local checkpoint at {args.finetune}")
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # 删除不匹配的层
        for k in ['head.l.weight', 'head.l.bias', 'head_dist.l.weight', 'head_dist.l.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # 加载权重到模型
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)
    print("model instantiated successfully.")

    # 设置模型 EMA
    model_ema = None
    if args.model_ema:
        # 在模型放入 GPU 后创建 EMA，设置 EMA 模型的衰减率
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            resume=''  # 如果需要从检查点恢复 EMA 状态，可以设置路径
        )

    # 计算模型参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 线性学习率缩放
    linear_scaled_lr = args.lr * args.batch_size / 512.0
    args.lr = linear_scaled_lr

    # 继续其他初始化（优化器、学习率调度器等）
    optimizer = create_optimizer(args, model)
    print(f"Optimizer created with parameters: {optimizer}")

    loss_scaler = NativeScaler()
    print("NativeScaler instantiated for mixed precision training.")

    lr_scheduler, _ = create_scheduler(args, optimizer)
    print(f"Learning rate scheduler created: {lr_scheduler}")

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        checkpoint = torch.load(args.teacher_path, map_location='cpu') # ,weights_only=True

        # 删除分类头的预训练权重（ImageNet 为 1000 类）
        state_dict = checkpoint['model']
        state_dict.pop('head.weight', None)
        state_dict.pop('head.bias', None)

        # 加载其余预训练权重
        teacher_model.load_state_dict(state_dict, strict=False)

        # 替换分类头为适配二分类的结构
        num_features = teacher_model.head.in_features
        teacher_model.head = nn.Linear(num_features, 2)  # 二分类输出为 2

        # 将模型转移到设备上
        teacher_model.to(device)


    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    # 检查并创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 记录模型结构和参数配置
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")

    # 加载检查点
    if args.resume:
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu',
                                                        check_hash=True) if args.resume.startswith(
            'https') else torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    # 设置模型 EMA
    model_ema = ModelEma(model, decay=args.model_ema_decay) if args.model_ema else None

    if args.eval:
        print("Starting evaluation...")
        test_stats = evaluate(data_loader_val, model, device, epoch=0, output_dir=output_dir)
        print(f"Accuracy on test images: {test_stats['acc1']:.2f}%")
        return

    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        progress_bar = tqdm(data_loader_train, desc=f'Training Epoch {epoch + 1}/{args.epochs}', leave=False)

        # Training step
        model.train()
        for samples, targets in progress_bar:
            samples, targets = samples.to(device), targets.to(device)

            # 应用 Mixup 数据增强
            if mixup_fn:
                samples, targets = mixup_fn(samples, targets)

            outputs = model(samples)
            loss = criterion(outputs, targets,inputs=samples) # 改变损失函数在不同蒸馏函数的使用下传回来是一个张量还是两个张量

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update model EMA
        if args.model_ema:
            model_ema.update(model)
        lr_scheduler.step(epoch)
        # Evaluation step
        model.eval()
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Validation accuracy after epoch {epoch + 1}: {test_stats['acc1']:.1f}%")

        # 检查并更新最佳准确度
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            # 保存最佳模型检查点
            checkpoint_path = output_dir / "checkpoint_best.pth"

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                checkpoint['model_ema'] = get_state_dict(model_ema)

            torch.save(checkpoint, checkpoint_path)
            print(f"New best model saved at {checkpoint_path} with accuracy: {max_accuracy:.2f}%")

        # 保存当前 epoch 的检查点
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                checkpoint['model_ema'] = get_state_dict(model_ema)

            torch.save(checkpoint, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training completed. Total training time: {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FAM_SFF training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = args.output_dir+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
