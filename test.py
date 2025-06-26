import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from model.GFFD.model_core import Two_Stream_Net
from model.GocNet.modelGocNet import GocNet
from model.MLFFE_ViT.MLFFE_ViT import MLDWT_ViT
from model.NPR.resnet import resnet50
from model.ViT.vit_models import ViT
from model.efficient.efficientnet_model import EfficientNet
from model.fcf import FCFNet
from data.datasets import build_dataset
from torch.utils.data import DataLoader
from main import get_args_parser  # 引入主函数的参数结构
from model.xception.xception_model import Xception


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs= model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    acc = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)

    # G-Mean 计算
    cm = confusion_matrix(all_targets, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = np.sqrt(sensitivity * specificity)
    else:
        gmean = 0.0

    print("\n===== Evaluation Results =====")
    print(f"Accuracy:  {acc:.4f}%")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"G-Mean:    {gmean:.4f}")

def main():
    # 获取完整参数结构并初始化默认值
    parser = get_args_parser()
    args = parser.parse_args([])  # 空列表表示不从命令行读取，而是用默认值 + 手动赋值

    # ===== 你只需要在这里设置参数值 =====
    args.data_path = 'datasets/CelebDFv2'
    args.data_set = 'CelebDFv2'
    args.num_classes = 2
    args.batch_size = 32
    args.decode_channels = 96
    args.dropout = 0.1
    args.window_size = 8
    args.device = 'cuda:0'
    args.num_workers = 4
    args.finetune = 'checkpoints/checkpoint_best.pth'
    args.eval = True
    # ====================================

    device = torch.device(args.device)

    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )

    model = FCFNet(
        decode_channels=args.decode_channels,
        num_classes=args.num_classes,
        dropout=args.dropout,
        window_size=args.window_size,
        distillation=args.distillation_type
    )

    # model = ViT(
    #     image_size=args.input_size,  # 图像尺寸 224
    #     patch_size=16,  # Patch 大小 16
    #     num_classes=args.num_classes,  # 类别数
    #     dim=768,  # Transformer token 维度
    #     depth=12,  # Transformer 深度
    #     heads=12,  # 注意力头数
    #     mlp_dim=3072,  # MLP 隐藏层大小
    #     pool='cls',  # 分类方式（使用 cls token）
    #     channels=3,  # RGB 3通道
    #     dim_head=64,  # 每个注意力头的维度
    #     dropout=args.dropout,  # dropout
    #     emb_dropout=args.dropout  # 位置编码 dropout
    # )

    # model = MLDWT_ViT(image_size=224, num_classes=2)

    # model = resnet50(pretrained=False, num_classes=2)

    # model = EfficientNet.from_name('efficientnet-b4', in_channels=3, num_classes=2)

    # model = Xception(num_classes=args.num_classes)

    # model = GocNet(num_classes=args.num_classes)

    # model = Two_Stream_Net()

    checkpoint = torch.load(args.finetune, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    evaluate(model, data_loader_val, device)

if __name__ == '__main__':
    main()
