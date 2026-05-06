"""
Three-Layer Dynamic Compensation Loss for Multi-Task ECG Diagnosis.

Implements:
1. Sample-level: Focal Loss for both multi-class and multi-label tasks
2. Class-level: Learned class weights with softmax normalization
3. Task-level: Homoscedastic uncertainty weighting (existing mechanism)

This creates a "pain of not learning" feedback loop where difficult classes
automatically get higher loss weights during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
    - p_t: model's estimated probability for the correct class
    - γ (gamma): focusing parameter (default=2)
    - α_t: class weight (optional)

    The (1-p_t)^γ term automatically down-weights easy examples (where p_t → 1)
    and focuses training on hard examples (where p_t is small).
    """

    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                class_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class indices for multi-class, or (B, C) for multi-label
            class_weights: (C,) optional class weights

        Returns:
            loss scalar
        """
        # Determine if multi-class or multi-label
        if targets.dim() == 1 or (targets.dim() == 2 and targets.shape[1] == 1):
            # Multi-class: targets are class indices
            t = targets.long()
            if t.dim() > 1:
                t = t.squeeze()
            return self._focal_multiclass(inputs, t, class_weights)
        else:
            # Multi-label: targets are binary vectors
            return self._focal_multilabel(inputs, targets.float(), class_weights)

    def _focal_multiclass(self, logits: torch.Tensor, targets: torch.Tensor,
                         class_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Focal loss for multi-class classification.

        Args:
            logits: (B, C) raw scores
            targets: (B,) class indices [0, C-1]
            class_weights: (C,) optional per-class weights
        """
        B, C = logits.shape

        # Compute log probability
        log_p = F.log_softmax(logits, dim=1)  # (B, C)

        # Gather log prob for target class
        targets_log_p = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Compute p_t (probability of true class)
        p_t = targets_log_p.exp()  # (B,)

        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma  # (B,)

        # Apply class weights if provided
        if class_weights is not None:
            weights = class_weights[targets]  # (B,)
            focal_weight = focal_weight * weights

        # Compute loss
        loss = -focal_weight * targets_log_p  # (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _focal_multilabel(self, logits: torch.Tensor, targets: torch.Tensor,
                         class_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Focal loss for multi-label classification (binary focal per label).

        Args:
            logits: (B, C) raw scores
            targets: (B, C) binary targets {0, 1}
            class_weights: (C,) optional per-class weights
        """
        # Compute sigmoid probability
        p_t = torch.sigmoid(logits)  # (B, C)

        # Binary cross entropy with logits
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B, C)

        # Focal term: for positive samples: (1 - p_t)^gamma, for negative: p_t^gamma
        pt = torch.where(targets == 1, p_t, 1 - p_t)
        focal_weight = (1 - pt) ** self.gamma  # (B, C)

        # Combine
        loss = focal_weight * bce  # (B, C)

        # Apply class weights if provided
        if class_weights is not None:
            loss = loss * class_weights.unsqueeze(0)  # (B, C)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LearnedClassWeights(nn.Module):
    """
    Learnable per-class weights for multi-task classification.

    Each task has its own weight vector of length (num_classes).
    Weights are initialized to 1.0 and learned via backpropagation.

    The key innovation: use softmax normalization so weights always
    average to 1.0, preventing loss explosion.
    """

    def __init__(
        self,
        task_config: Dict[str, Tuple[int, str]],
        device: torch.device,
        static_class_weights: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            task_config: {task_name: (num_classes, task_type)}
                         task_type: 'mc' for multi-class, 'ml' for multi-label
            device: torch device
        """
        super().__init__()
        self.task_config = task_config
        self.device = device
        self.static_class_weights = static_class_weights or {}

        # Create learnable weights for each task
        # Use regular dict and register_parameter manually to handle "." in names
        self._weight_names = {}  # Maps task_name -> parameter name (safe name)

        for task_name, (num_classes, task_type) in task_config.items():
            # Create a safe parameter name (replace "." with "_")
            safe_name = task_name.replace(".", "_")
            # Initialize to zeros (softmax will give uniform weights)
            weight = nn.Parameter(torch.zeros(num_classes, device=device))
            self.register_parameter(safe_name, weight)
            self._weight_names[task_name] = safe_name

            static_weight = self.static_class_weights.get(task_name)
            if static_weight is not None:
                static_weight = static_weight.to(device=device, dtype=torch.float32)
                if static_weight.numel() != num_classes:
                    raise ValueError(
                        f"Static class weight size mismatch for {task_name}: "
                        f"expected {num_classes}, got {static_weight.numel()}"
                    )
                self.register_buffer(f"{safe_name}_static", static_weight)

        # Store for easy access
        self.task_names = list(task_config.keys())

    def get_weights(self, task_name: str) -> torch.Tensor:
        """
        Get normalized weights for a task.

        Uses softmax to ensure:
        - All weights are positive
        - Mean weight = 1.0 (we multiply by num_classes)
        - No single weight can dominate completely

        Args:
            task_name: name of the task

        Returns:
            (num_classes,) tensor of normalized weights
        """
        if task_name not in self._weight_names:
            return None

        safe_name = self._weight_names[task_name]
        raw_weights = getattr(self, safe_name)  # (num_classes,)

        # Softmax normalization + scale by num_classes
        # This ensures weights average to 1.0
        normalized = F.softmax(raw_weights, dim=0) * len(raw_weights)
        static_name = f"{safe_name}_static"
        if hasattr(self, static_name):
            normalized = normalized * getattr(self, static_name)
            normalized = normalized / normalized.mean().clamp(min=1e-6)

        return normalized

    def get_weight_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about learned weights for logging.

        Returns:
            {task_name: {'min': float, 'max': float, 'std': float, 'values': list}}
        """
        stats = {}
        for task_name in self.task_names:
            w = self.get_weights(task_name)
            if w is not None:
                stats[task_name] = {
                    'min': float(w.min().item()),
                    'max': float(w.max().item()),
                    'std': float(w.std().item()),
                    'values': w.cpu().tolist()
                }
        return stats

    def forward(self, task_name: str) -> torch.Tensor:
        """Convenience method to get weights for a task."""
        return self.get_weights(task_name)


class DynamicMultiTaskLoss(nn.Module):
    """
    Three-layer dynamic compensation loss for multi-task ECG diagnosis.

    Layer 1 (Sample-level): Focal Loss - down-weights easy examples
    Layer 2 (Class-level): Learned Class Weights - up-weights difficult classes
    Layer 3 (Task-level): Uncertainty Weighting - balances task gradients
    """

    def __init__(
        self,
        task_config: Dict[str, Tuple[int, str]],
        focal_gamma: float = 2.0,
        device: torch.device = torch.device('cuda:0'),
        init_log_vars: float = 0.0,
        static_class_weights: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            task_config: {task_name: (num_classes, task_type)}
            focal_gamma: focusing parameter for Focal Loss
            device: torch device
            init_log_vars: initial value for uncertainty log_vars
        """
        super().__init__()

        self.task_config = task_config
        self.focal_gamma = focal_gamma
        self.device = device

        # Layer 1: Focal Loss
        self.focal_loss = FocalLoss(gamma=focal_gamma, reduction='none')

        # Layer 2: Learned Class Weights
        self.class_weights = LearnedClassWeights(task_config, device, static_class_weights)

        # Layer 3: Learnable task uncertainties (homoscedastic)
        # One log_var per task, initialized to 0 (equal weighting)
        num_tasks = len(task_config)
        self.register_parameter(
            'log_vars',
            nn.Parameter(torch.full((num_tasks,), init_log_vars, device=device))
        )

        self.task_names = list(task_config.keys())
        self.task_types = {name: typ for name, (_, typ) in task_config.items()}

    def compute_task_loss(
        self,
        task_name: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss for a single task with all three layers applied.

        Args:
            task_name: name of the task
            logits: model predictions
            targets: ground truth labels
            log_var: uncertainty log variance for this task

        Returns:
            (weighted_loss, details_dict)
        """
        # Layer 2: Get learned class weights
        class_weights = self.class_weights(task_name)

        # Layer 1: Compute Focal Loss (unreduced, per-sample)
        if self.task_types[task_name] == 'mc':
            # Multi-class: targets should be class indices
            if targets.dim() > 1:
                targets = targets.squeeze()
            per_sample_loss = self.focal_loss(logits, targets.long(), class_weights)
        else:
            # Multi-label: targets are binary vectors
            per_sample_loss = self.focal_loss(logits, targets.float(), class_weights)

        # Layer 3: Apply uncertainty weighting
        # Formula: L = 1/(2*exp(log_var)) * focal_loss + log_var/2
        precision = torch.exp(log_var).clamp(max=1e6)
        weighted_loss = per_sample_loss / (2.0 * precision) + log_var / 2.0

        # Reduce to scalar (mean over batch)
        weighted_loss = weighted_loss.mean()

        # Gather details for logging
        details = {
            'raw_loss': per_sample_loss.detach().mean().item(),
            'weighted_loss': weighted_loss.detach().item(),
            'precision': precision.detach().item(),
            'class_weights': class_weights.detach() if class_weights is not None else None,
        }

        return weighted_loss, details

    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor],
        label_map: Dict[str, Tuple[str, int, str]],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total multi-task loss with three-layer dynamic compensation.

        Args:
            logits_dict: {task_name: logits} from model
            labels_dict: {group_name: labels} from dataset
            label_map: maps task_name to (label_group, idx, task_type)

        Returns:
            (total_loss, details_dict)
        """
        total_loss = torch.tensor(0.0, device=self.device)
        reg_loss = torch.tensor(0.0, device=self.device)

        details = {}

        for i, task_name in enumerate(self.task_names):
            if task_name not in logits_dict:
                continue

            # Get logits and targets
            logits = logits_dict[task_name]
            label_group, idx, _ = label_map[task_name]
            targets = labels_dict[label_group].to(self.device)

            # Extract the right index if multi-task group
            if targets.dim() > 1 and targets.shape[1] > 1:
                targets = targets[:, idx]

            # Get log_var for this task
            log_var = self.log_vars[i]

            # Compute task loss with all three layers
            task_loss, task_details = self.compute_task_loss(
                task_name, logits, targets, log_var
            )

            total_loss = total_loss + task_loss
            reg_loss = reg_loss + log_var / 2.0
            details[task_name] = task_details

        details['total'] = total_loss.detach().item()
        details['regularization'] = reg_loss.detach().item()

        return total_loss, details

    def get_log_stats(self) -> Dict[str, float]:
        """Get log statistics for TensorBoard logging."""
        stats = {}

        # Class weights stats
        class_weight_stats = self.class_weights.get_weight_stats()
        for task_name, task_stats in class_weight_stats.items():
            stats[f'class_weights/{task_name}'] = task_stats['values']
            stats[f'class_weights_std/{task_name}'] = task_stats['std']
            stats[f'class_weights_range/{task_name}'] = task_stats['max'] - task_stats['min']

        # Task uncertainty stats
        for i, task_name in enumerate(self.task_names):
            log_var = self.log_vars[i].item()
            stats[f'log_var/{task_name}'] = log_var
            stats[f'weight/{task_name}'] = 0.5 / torch.exp(torch.tensor(log_var)).item()

        return stats
