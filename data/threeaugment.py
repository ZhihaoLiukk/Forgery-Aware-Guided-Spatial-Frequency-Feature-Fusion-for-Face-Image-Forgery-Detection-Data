"""
3Augment 实现，适用于人脸篡改检测。
数据增强包括颜色抖动、随机裁剪、以及高斯模糊等。
"""
import torch
from torchvision import transforms
from PIL import ImageFilter, ImageOps
import random

class GaussianBlur(object):
    """
    应用高斯模糊到 PIL 图像。
    """
    def __init__(self, p=0.3, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() <= self.prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
        return img

class Solarization(object):
    """
    对 PIL 图像应用反色效果。
    """
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

def new_data_aug_generator(args=None):
    """
    创建适用于人脸篡改检测的数据增强方法。
    """
    img_size = args.input_size if args else 224
    primary_tfl = [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip()
    ]

    secondary_tfl = [
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        GaussianBlur(p=0.2),
        Solarization(p=0.1)
    ]

    final_tfl = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # 将主、次和最终增强方法组合起来
    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
