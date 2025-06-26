import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import cv2


class ExpressionMLP(nn.Module):
    def __init__(self, in_channels=56, hidden=32, num_classes=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.fc(self.pool(x))


class FAM_GS(nn.Module):
    """
    Forgery-Aware Module Guided by Facial Symmetry and Semantic Consistency (FAM-GS)
    """
    def __init__(self, in_channels=288, num_expression_classes=8, input_size=224):
        super(FAM_GS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 112, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(112, 56, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(56, 56, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.mlp_global = ExpressionMLP(56, hidden=32, num_classes=num_expression_classes)
        self.mlp_local = ExpressionMLP(56, hidden=32, num_classes=num_expression_classes)
        self.input_size = input_size

        # Mediapipe 初始化（用于人脸关键点）
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    def extract_keypoints(self, image_tensor):
        """
        使用 Mediapipe 提取图像中人脸关键点（支持 B=1），返回左眼、右眼、嘴角框坐标。
        """
        if image_tensor.shape[0] != 1:
            return None

        # 转为 BGR numpy 图像
        image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        image_np = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        results = self.mp_face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = self.input_size, self.input_size

        def get_bbox(ids, margin=5):
            xs = [int(landmarks[i].x * w) for i in ids]
            ys = [int(landmarks[i].y * h) for i in ids]
            x1, x2 = max(min(xs) - margin, 0), min(max(xs) + margin, w)
            y1, y2 = max(min(ys) - margin, 0), min(max(ys) + margin, h)
            return (x1, y1, x2, y2)

        return {
            'left_eye': get_bbox([33, 133]),
            'right_eye': get_bbox([362, 263]),
            'mouth_left': get_bbox([61]),
            'mouth_right': get_bbox([291]),
        }

    def map_coords(self, kp_dict, feat_h, feat_w):
        """
        将输入图像 (224x224) 中的关键点坐标映射到特征图尺寸。
        """
        scale_x = feat_w / self.input_size
        scale_y = feat_h / self.input_size
        mapped = {}
        for k, (x1, y1, x2, y2) in kp_dict.items():
            mapped[k] = (
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y)
            )
        return mapped

    def forward(self, x, input_image=None, with_symmetry=True):
        """
        Args:
            x: 特征图 [B, C, H, W]
            input_image: 输入图像 (224x224)，仅支持 B=1，用于提取人脸关键点
            with_symmetry: 是否启用对称结构损失
        Returns:
            m: 伪造显著性图 [B, 1, H, W]
            loss_sym: 结构一致性损失
            loss_exp: 表情一致性损失
            loss_sem_sym: 区域语义对称性损失
        """
        feat = self.encoder(x)
        m = self.decoder(feat)

        loss_sym = 0.0
        loss_exp = 0.0
        loss_sem_sym = 0.0

        # --- 结构对称性损失 ---
        if with_symmetry:
            x_flip = torch.flip(x, dims=[3])
            feat_flip = self.encoder(x_flip)
            feat_flip = torch.flip(feat_flip, dims=[3])
            loss_sym = F.l1_loss(feat, feat_flip)

        # --- 语义一致性建模 ---
        if input_image is not None:
            kp_dict_raw = self.extract_keypoints(input_image)
            if kp_dict_raw:
                feat_h, feat_w = feat.shape[-2:]
                kp_dict = self.map_coords(kp_dict_raw, feat_h, feat_w)

                # 全局表情预测
                pred_global = self.mlp_global(feat.detach())

                for region_name in kp_dict:
                    x1, y1, x2, y2 = kp_dict[region_name]
                    region_feat = F.adaptive_avg_pool2d(feat[:, :, y1:y2, x1:x2], 1)
                    pred_local = self.mlp_local(region_feat)
                    loss_exp += F.cross_entropy(pred_local, pred_global.argmax(dim=1))

                # 对称区域语义相似性损失
                if 'left_eye' in kp_dict and 'right_eye' in kp_dict:
                    x1, y1, x2, y2 = kp_dict['left_eye']
                    le = F.adaptive_avg_pool2d(feat[:, :, y1:y2, x1:x2], 1)
                    x1, y1, x2, y2 = kp_dict['right_eye']
                    re = F.adaptive_avg_pool2d(feat[:, :, y1:y2, x1:x2], 1)
                    loss_sem_sym += F.l1_loss(le, re)

                if 'mouth_left' in kp_dict and 'mouth_right' in kp_dict:
                    x1, y1, x2, y2 = kp_dict['mouth_left']
                    ml = F.adaptive_avg_pool2d(feat[:, :, y1:y2, x1:x2], 1)
                    x1, y1, x2, y2 = kp_dict['mouth_right']
                    mr = F.adaptive_avg_pool2d(feat[:, :, y1:y2, x1:x2], 1)
                    loss_sem_sym += F.l1_loss(ml, mr)

        return m, loss_sym, loss_exp, loss_sem_sym
