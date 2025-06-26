import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    该模块包装了标准的损失函数，并通过教师模型的预测添加了额外的知识蒸馏损失，作为辅助监督信号。
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        """
        初始化蒸馏损失模块。

        参数:
        - base_criterion: 基础损失函数，通常是交叉熵损失，用于计算学生模型的基本损失。
        - teacher_model: 教师模型，提供了软标签或硬标签作为额外的监督信号。
        - distillation_type: 蒸馏类型，可以是 'none'、'soft' 或 'hard'。
        - alpha: 蒸馏损失的权重，控制基础损失与蒸馏损失的权重分配。
        - tau: 温度参数，用于软蒸馏，平滑 softmax 分布。
        """
        super().__init__()
        self.base_criterion = base_criterion  # 基础损失函数
        self.teacher_model = teacher_model    # 教师模型
        # 确保 distillation_type 在定义的类型中
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type  # 蒸馏类型
        self.alpha = alpha  # 蒸馏损失的权重
        self.tau = tau      # 温度参数

    def forward(self, outputs, labels, inputs=None):
        """
        前向计算过程。计算基础损失并根据蒸馏类型选择是否添加蒸馏损失。

        参数:
        - outputs: 学生模型的输出，可以是一个张量或包含两个张量的元组，
                   第一个张量为原始输出，第二个张量为用于蒸馏的输出。
        - labels: 用于基础损失函数计算的标签。
        - inputs: 输入到教师模型的数据，用于获取教师模型的预测输出（仅用于蒸馏）。

        返回:
        - loss: 综合的损失，包含基础损失和蒸馏损失（如果启用了知识蒸馏）。
        """
        outputs_kd = None  # 初始化用于蒸馏的输出
        extra_losses = None
        if not isinstance(outputs, torch.Tensor):

            if len(outputs) == 2:
                outputs, outputs_kd = outputs
            elif len(outputs) == 3:
                outputs, outputs_kd, extra_losses = outputs
            else:
                raise ValueError(
                    "模型输出格式错误，应为 (logits,) 或 (logits, logits_kd) 或 (logits, logits_kd, extra_losses)")
            # 如果输出是元组，提取 outputs 和 outputs_kd
            # outputs, outputs_kd = outputs
         # 计算基础损失
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            final_loss = base_loss
        else:
            if outputs_kd is None:
                raise ValueError("当启用知识蒸馏时，模型应返回一个 Tuple 包含蒸馏输出")

            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)

            if self.distillation_type == 'soft':
                T = self.tau
                distillation_loss = F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (T * T) / outputs_kd.numel()
            elif self.distillation_type == 'hard':
                distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

            final_loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        # === 额外引导损失 ===
        if extra_losses is not None:

            print("[DEBUG - Inside DistillationLoss] extra_losses keys:", extra_losses.keys())
            print("[DEBUG - Inside DistillationLoss] type of loss_sym:", type(extra_losses["loss_sym"]))
            print("[DEBUG - Inside DistillationLoss] shape or content:", extra_losses["loss_sym"])

            if "loss_sym" in extra_losses:
                final_loss += 0.1 * extra_losses["loss_sym"].mean()
            if "loss_exp" in extra_losses:
                final_loss += 0.2 * extra_losses["loss_exp"].mean()
            if "loss_sem_sym" in extra_losses:
                final_loss += 0.2 * extra_losses["loss_sem_sym"].mean()

        return final_loss  # 返回综合损失

