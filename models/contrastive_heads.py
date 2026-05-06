"""
分类头对比学习模块 - 即插即用组件

在每个分类头中加入对比学习，使得同类样本特征更接近，异类样本特征更远离。
支持多分类和多标签任务。
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadContrastiveLoss(nn.Module):
    """
    分类头级别的监督对比学习损失

    核心思想：对于每个分类头，同类样本的特征应该更接近，异类样本的特征应该更远离。

    支持两种模式：
    1. multi_class: 多分类任务，每个样本属于一个类别
    2. multi_label: 多标签任务，每个样本可能属于多个类别

    Args:
        temperature: 温度参数，控制对比学习的强度（越小越严格）
        mode: 'multi_class' 或 'multi_label'
        max_samples_per_class: 每个类别最多使用的样本数，避免内存溢出
    """

    def __init__(
        self,
        temperature: float = 0.1,
        mode: str = 'multi_class',
        max_samples_per_class: int = 64,
    ):
        super().__init__()
        self.temperature = temperature
        self.mode = mode
        self.max_samples_per_class = max_samples_per_class

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算监督对比学习损失

        Args:
            features: (B, D) 特征向量
            labels: (B,) 或 (B, num_classes) 标签
                    - multi_class: (B,) 类别索引
                    - multi_label: (B, num_classes) 二值标签
            mask: (B,) 有效样本掩码（用于处理无效标签）

        Returns:
            对比学习损失标量
        """
        B = features.size(0)
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        device = features.device

        # 处理掩码
        if mask is not None:
            valid_idx = mask.nonzero(as_tuple=True)[0]
            if len(valid_idx) < 2:
                return torch.tensor(0.0, device=features.device)
            features = features[valid_idx]
            labels = labels[valid_idx]
            B = features.size(0)

        # L2 归一化
        features = F.normalize(features, dim=-1)

        # 计算相似度矩阵
        # sim_matrix[i, j] = cos_sim(features[i], features[j]) / temperature
        sim_matrix = torch.matmul(features, features.t()) / self.temperature

        if self.mode == 'multi_class':
            return self._multi_class_loss(sim_matrix, labels, device)
        else:
            return self._multi_label_loss(sim_matrix, labels, device)

    def _multi_class_loss(
        self,
        sim_matrix: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        多分类任务的监督对比学习损失

        对于每个样本 i：
        - 正样本：与 i 同类别的其他样本
        - 负样本：与 i 不同类别的样本

        Loss = - log(exp(sim(i, pos)) / sum(exp(sim(i, all))))
        """
        B = sim_matrix.size(0)

        # 创建标签掩码：mask[i, j] = 1 如果 labels[i] == labels[j]
        # (B, B)
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        label_mask = label_mask.float().to(sim_matrix.device)  # 确保在同一设备

        # 移除对角线（自己和自己）
        logits_mask = torch.ones_like(label_mask)
        logits_mask.fill_diagonal_(0)

        # 对于每个样本，计算其正样本和负样本的相似度
        # 只考虑有足够正样本的类别
        class_counts = label_mask.sum(dim=1)

        # 计算损失
        loss = 0.0
        valid_count = 0

        for i in range(B):
            # 跳过正样本太少的情况
            if class_counts[i] <= 1:
                continue

            # 正样本掩码（不包括自己）
            pos_mask = label_mask[i] * logits_mask[i]
            num_pos = pos_mask.sum().item()

            if num_pos == 0:
                continue

            # 正样本相似度
            pos_sim = sim_matrix[i][pos_mask > 0]

            # 负样本掩码
            neg_mask = (1 - label_mask[i]) * logits_mask[i]
            neg_sim = sim_matrix[i][neg_mask > 0]

            # 如果没有负样本，跳过
            if neg_sim.numel() == 0:
                continue

            # 计算对比损失
            # Loss = -log(sum(exp(pos_sim)) / (sum(exp(pos_sim)) + sum(exp(neg_sim))))
            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim)

            loss += -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))
            valid_count += 1

        return loss / max(valid_count, 1)

    def _multi_label_loss(
        self,
        sim_matrix: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        多标签任务的监督对比学习损失

        对于每个样本 i：
        - 正样本：与 i 共享至少一个标签的样本
        - 负样本：与 i 没有共同标签的样本

        使用软权重：共享标签越多，正样本权重越大
        """
        B = sim_matrix.size(0)

        # 计算标签重叠矩阵
        # overlap[i, j] = labels[i] 和 labels[j] 共享的标签数
        # (B, B)
        overlap = torch.matmul(labels.float(), labels.t().float())

        # 正样本权重：共享标签数 / 标签数的几何平均
        label_counts = labels.sum(dim=1).unsqueeze(1)  # (B, 1)
        normalizer = torch.sqrt(label_counts * label_counts.t()) + 1e-8
        pos_weights = overlap / normalizer

        # 移除对角线
        logits_mask = torch.ones_like(pos_weights)
        logits_mask.fill_diagonal_(0)
        pos_weights = pos_weights * logits_mask

        # 负样本掩码：没有共同标签的样本
        neg_mask = (overlap == 0) & (logits_mask.bool())

        loss = 0.0
        valid_count = 0

        for i in range(B):
            # 正样本
            pos_mask = pos_weights[i] > 0
            num_pos = pos_mask.sum().item()

            if num_pos == 0:
                continue

            # 加权正样本相似度 - 确保在同一设备上
            pos_sim = sim_matrix[i][pos_mask]
            pos_w = pos_weights[i][pos_mask].to(pos_sim.device)  # 修复：确保在同一设备
            weighted_pos_sim = pos_sim * pos_w

            # 负样本
            neg_sim = sim_matrix[i][neg_mask[i]]

            if neg_sim.numel() == 0:
                continue

            # 计算损失
            pos_exp = torch.exp(weighted_pos_sim)
            neg_exp = torch.exp(neg_sim)

            loss += -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))
            valid_count += 1

        return loss / max(valid_count, 1)


class MultiHeadContrastive(nn.Module):
    """
    多头对比学习包装器 - 即插即用组件

    为每个分类头单独计算对比学习损失，然后加权平均。

    Args:
        head_names: 各个头的名称列表
        temperature: 温度参数
        head_modes: 每个头的模式 {'head_name': 'multi_class' 或 'multi_label'}
        weights: 每个头的对比损失权重
    """

    def __init__(
        self,
        head_names: list,
        temperature: float = 0.1,
        head_modes: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.head_names = head_names

        # 默认所有头都是 multi_class
        head_modes = head_modes or {}
        self.head_modes = {name: head_modes.get(name, 'multi_class') for name in head_names}

        # 默认所有头权重为 1.0
        weights = weights or {}
        self.weights = {name: weights.get(name, 1.0) for name in head_names}

        # 为每个头创建对比学习模块
        self.contrastive_modules = nn.ModuleDict({
            name: HeadContrastiveLoss(
                temperature=temperature,
                mode=self.head_modes[name],
            )
            for name in head_names
        })

    def forward(
        self,
        head_features: Dict[str, torch.Tensor],
        head_labels: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有头的对比学习损失

        Args:
            head_features: {head_name: (B, D)} 特征字典
            head_labels: {head_name: labels} 标签字典
            masks: {head_name: (B,)} 有效样本掩码字典

        Returns:
            {head_name: loss} 每个头的对比损失字典
        """
        losses = {}
        masks = masks or {}

        for name in self.head_names:
            if name not in head_features or name not in head_labels:
                continue

            features = head_features[name]
            labels = head_labels[name]
            mask = masks.get(name, None)

            loss = self.contrastive_modules[name](features, labels, mask)
            losses[name] = loss * self.weights[name]

        return losses

    def get_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取加权总损失"""
        if not losses:
            return torch.tensor(0.0)
        return sum(losses.values())


class PrototypicalContrastive(nn.Module):
    """
    基于原型的对比学习 - 适用于类别不平衡的情况

    为每个类别维护一个原型向量，样本与同类原型的距离应该更近。
    这样可以缓解类别不平衡问题，因为原型是动态更新的平均值。

    Args:
        num_classes: 类别数量
        feature_dim: 特征维度
        temperature: 温度参数
        momentum: 原型更新的动量系数
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        temperature: float = 0.1,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.momentum = momentum

        # 注册原型向量（使用 buffer 而不是 parameter，因为会手动更新）
        self.register_buffer(
            'prototypes',
            F.normalize(torch.randn(num_classes, feature_dim), dim=-1)
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        update_prototypes: bool = True,
    ) -> torch.Tensor:
        """
        计算基于原型的对比学习损失

        Args:
            features: (B, D) 特征向量
            labels: (B,) 类别索引
            update_prototypes: 是否更新原型

        Returns:
            对比学习损失
        """
        B = features.size(0)
        if B == 0:
            return torch.tensor(0.0, device=features.device)

        device = features.device

        # L2 归一化
        features_norm = F.normalize(features, dim=-1)

        # 计算与原型的相似度
        # (B, num_classes)
        sim_to_prototypes = torch.matmul(features_norm, self.prototypes.t()) / self.temperature

        # 交叉熵损失
        loss = F.cross_entropy(sim_to_prototypes, labels)

        # 更新原型（在训练时）
        if update_prototypes and self.training:
            with torch.no_grad():
                for c in range(self.num_classes):
                    mask = labels == c
                    if mask.sum() > 0:
                        class_features = features_norm[mask]
                        # 计算类别均值
                        class_mean = class_features.mean(dim=0)
                        class_mean = F.normalize(class_mean, dim=0)
                        # 动量更新
                        self.prototypes[c] = (
                            self.momentum * self.prototypes[c] +
                            (1 - self.momentum) * class_mean
                        )
                        self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)

        return loss
