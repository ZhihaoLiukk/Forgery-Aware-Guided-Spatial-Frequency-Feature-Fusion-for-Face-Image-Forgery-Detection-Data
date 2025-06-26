import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from .spatialloacl import SppCSPC


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

# 深度可分离卷积模块
class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True,
                 dynamic_window=False):
        super().__init__()
        self.num_heads = num_heads
        self.dynamic_window = dynamic_window
        self.base_window_size = window_size  # 默认的基础窗口大小
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads

        if head_dim == 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) cannot be larger than dim ({dim}). Please adjust num_heads or dim.")
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size) #局部卷积增强（proj）：使用深度可分离卷积（SeparableConvBN）在窗口级别增强特征。

        self.attn_x = nn.Conv2d(dim,dim,kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.Conv2d(dim,dim,kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        if self.relative_pos_embedding:
            self._init_relative_position(window_size)


    def _init_relative_position(self, window_size):
        # 初始化相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        # 动态调整窗口大小
        if self.dynamic_window:
            self.ws = max(4, min(H, W) // 8)  # 根据图像尺寸动态调整窗口大小
        else:
            self.ws = self.base_window_size

        if self.relative_pos_embedding and (self.relative_position_index.shape[0] != self.ws * self.ws):
            self._init_relative_position(self.ws)  # 重新初始化相对位置编码

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1) @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class LocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 window_size=8,
                 dynamic_window=False  # 是否启用动态窗口机制
                 ):
        super().__init__()

        self.dynamic_window = dynamic_window
        self.base_window_size = window_size  # 默认的基础窗口大小
        self.local = SppCSPC(dim, dim)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def adjust_window_size(self, x):
        """动态调整窗口大小"""
        B, C, H, W = x.shape

        # 根据输入特征图的尺寸动态调整窗口大小
        if self.dynamic_window:
            new_window_size = max(4, min(H, W) // 8)  # 动态计算窗口大小，最小为 4
            return new_window_size
        return self.base_window_size  # 使用默认窗口大小

    def forward(self, x):
        B, C, H, W = x.shape

        # 动态调整窗口大小
        current_window_size = self.adjust_window_size(x)

        # 如果窗口大小发生变化，更新卷积核大小
        if hasattr(self, 'ws') and self.ws != current_window_size:
            self.ws = current_window_size
            self.proj = SeparableConvBN(C, C, kernel_size=self.ws)

        elif not hasattr(self, 'ws'):  # 首次初始化
            self.ws = current_window_size

        # 局部特征提取
        local = self.local(x)

        # 卷积增强 + 裁剪
        out = self.pad_out(local)
        out = self.proj(out)
        out = out[:, :, :H, :W]  # 裁剪至原始尺寸

        return out


class LocalBlock(nn.Module):
    expansion = 1
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn =LocalAttention(dim,window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class multilocalBlock(nn.Module):
    expansion = 1
    def __init__(self,dim=256,outdim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.down = Conv(dim,outdim,kernel_size=3,stride=2,dilation=1,bias=False)
        self.norm1 = norm_layer(outdim)
        self.attn =LocalAttention(outdim,window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=outdim, hidden_features=mlp_hidden_dim, out_features=outdim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(outdim)

    def forward(self, x):
        x = self.down(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.drop_path(self.norm2(x))

        return x
class GlobalBlock(nn.Module):
    expansion = 1
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
class GlBlock(nn.Module):
    expansion = 1
    def __init__(self, dim=256,outdim = 256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.down = Conv(dim, outdim, kernel_size=3, stride=2, dilation=1, bias=False)
    def forward(self, x):
        x = self.down(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.norm2(x)

        return x



class SWT(nn.Module):
    def __init__(self, in_ch, out_ch,num_heads=8, window_size=8):
        super(SWT, self).__init__()
        self.glb = GlBlock(dim=in_ch,outdim=in_ch,num_heads=num_heads, window_size=window_size)
        self.localb=multilocalBlock(dim=in_ch,outdim=in_ch,num_heads=8, window_size=window_size)
        self.outconv_bn_relu_glb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_local = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x,imagename=None):

        glb = self.outconv_bn_relu_glb(self.glb(x))
        local = self.outconv_bn_relu_local(self.localb(x))
        return glb,local

