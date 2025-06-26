import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import torch.nn.functional as F


class DWT(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(DWT, self).__init__()
        # 使用DWTForward进行一次小波变换
        self.wt = DWTForward(J=1, mode='zero', wave='db1')
        # 第二次小波变换的配置
        self.wt2 = DWTForward(J=1, mode='zero', wave='db1')  # 可以修改为其他小波函数或配置

        # 第一部分卷积、批归一化和ReLU
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        # 输出卷积层：低频部分（LL2）
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # 输出卷积层：高频部分（HL2, LH2, HH2）
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.outconv_bn_relu_H2 = nn.Sequential(
            nn.Conv2d(4*in_ch, out_ch, kernel_size=1, stride=1),  # 修改输入通道数为 9
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 第一次小波变换
        yL1, yH1 = self.wt(x)
        y_HL1 = yH1[0][:, :, 0, ::]
        y_LH1 = yH1[0][:, :, 1, ::]
        y_HH1 = yH1[0][:, :, 2, ::]

        yH1 = torch.cat([y_HL1, y_LH1, y_HH1], dim=1)
        yH1 = self.conv_bn_relu(yH1)

        # 第二次小波变换，作用在第一次变换的低频部分LL1上
        yL2, yH2 = self.wt2(yL1)
        y_HL2 = yH2[0][:, :, 0, ::]
        y_LH2 = yH2[0][:, :, 1, ::]
        y_HH2 = yH2[0][:, :, 2, ::]

        yH2 = torch.cat([y_HL2, y_LH2, y_HH2], dim=1)
        yH1h, yH1w = yH1.size()[-2:]
        yH2 = F.interpolate(yH2, size=(yH1h, yH1w), mode='bicubic', align_corners=False)

        yH2 = torch.cat([yH1, yH2], dim=1)

        # 对第二次小波变换的低频部分（LL2）进行处理
        yL2 = self.outconv_bn_relu_L(yL2)
        yL2 = F.interpolate(yL2, size=(yH1h, yH1w), mode='bicubic', align_corners=False)

        # 对第二次小波变换的高频部分（HL2, LH2, HH2）进行处理
        yH2 = self.outconv_bn_relu_H2(yH2)

        yL = yL2
        yH = yH2

        return yL, yH

