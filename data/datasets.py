import os
import json
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class FaceForgeryDataset(torch.utils.data.Dataset):
    """
    用于人脸篡改检测的数据集类。
    """

    def __init__(self, root, transform=None):
        """
        初始化数据集
        root: 数据集划分路径，如 train、test 或 validation 文件夹的路径
        transform: 图像增强方法
        """
        self.transform = transform
        self.samples = []

        # 加载子文件夹下的 annotations.json 文件
        annotations_path = os.path.join(root, 'annotations.json')  # <-- 每个子文件夹的 annotations.json 文件路径
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"{annotations_path} 文件未找到，请检查路径是否正确。")

        with open(annotations_path, 'r') as f:
            data = json.load(f)
            for entry in data["images"]:
                # 将图像路径和标签添加到样本列表中
                self.samples.append((os.path.join(root, entry["path"]), entry["label"]))

    def __getitem__(self, index):
        """
        根据索引返回图像及其标签
        """
        path, label = self.samples[index]
        # print(f"Sample: {path}, Label: {label}")
        try:
            image = default_loader(path)
        except Exception as e:
            print(f"Error loading image at {path}: {e}")
            return None  # 或者返回一个空样本或跳过该样本
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)



def build_dataset(is_train, args):
    """
    构建数据集函数
    is_train: 是否为训练模式
    args: 参数对象，包含数据路径、图像尺寸等信息
    """
    # 根据训练或验证选择相应子文件夹
    split = 'train' if is_train else 'validation'
    data_path = os.path.join(args.data_path, split)  # 指向 train 或 validation 文件夹
    transform = build_transform(is_train, args)

    # 数据集类型检查
    if args.data_set == 'your dataset':
        dataset = FaceForgeryDataset(data_path, transform=transform)  # <-- 传入子文件夹路径
        nb_classes = 2  # 假设二分类：真实和篡改
    else:
        raise ValueError(f"Dataset {args.data_set} is not recognized.")

    return dataset, nb_classes


def build_transform(is_train, args):
    """
    构建图像增强函数
    """
    resize_im = args.input_size > 32
    if is_train:
        # 训练时使用数据增强策略
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )

        if not resize_im:
            # 如果输入尺寸较小，不调整尺寸
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    # 测试时的图像增强（包括尺寸调整和归一化）
    t = []
    if args.finetune:
        t.append(transforms.Resize((args.input_size, args.input_size), interpolation=3))
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(transforms.Resize(size, interpolation=3))
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
