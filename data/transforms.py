"""
Signal & image transforms — aligned with GEM-main/ecg_coca/open_clip/.

Normalization constants:
  MIMIC-IV ECG:  zero-mean, unit-variance per lead  (mean=[0]*12, std=[1]*12)
  Image:         CLIP ViT-L/14@336px — OpenAI CLIP mean/std, resize->336, center-crop->336

Signal augmentations (from GEM ecg_coca/open_clip/augmentations/):
  - BaselineWander : p=0.5, max_amplitude=0.5, k=3 harmonics, fs=500
  - CutMix         : p=0.5, alpha=0.5
  - RandomMasking  : p=0.3, mask_width=[0.08, 0.18]s, fs=500

Image augmentation uses imgaug (identical to GEM gem_generation/ImageAugmentation/augment.py):
  - Affine rotation   : ±25°
  - Gaussian noise    : scale=25
  - Random crop       : 1%
  - Color temperature : 6500K
"""

import random
from copy import deepcopy
from numbers import Real
from random import shuffle as random_shuffle
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Constants  (from GEM ecg_coca/open_clip/constants.py)
# ---------------------------------------------------------------------------
MIMIC_IV_MEAN = [0.0] * 12
MIMIC_IV_STD  = [1.0] * 12

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD  = (0.26862954, 0.26130258, 0.27577711)

IMAGE_SIZE = 336  # ViT-L/14@336px

# ---------------------------------------------------------------------------
# Signal transforms
# ---------------------------------------------------------------------------

class Normalize(nn.Module):
    """Per-lead normalization (identical to GEM open_clip/ecg_transform.py)."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        self.mean = torch.tensor(mean if mean is not None else MIMIC_IV_MEAN, dtype=torch.float32)
        self.std  = torch.tensor(std  if std  is not None else MIMIC_IV_STD,  dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 12, seq_len)
        self.mean = self.mean.to(x.device)
        self.std  = self.std.to(x.device)
        for i in range(len(self.mean)):
            x[:, i, :] = (x[:, i, :] - self.mean[i]) / self.std[i]
        return x


class Resize(nn.Module):
    """Truncate / zero-pad to a fixed sequence length (identical to GEM Resize)."""

    def __init__(self, seq_length: int = 5000):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, length = x.shape
        if length < self.seq_length:
            new_x = torch.zeros((b, c, self.seq_length), dtype=x.dtype, device=x.device)
            new_x[:, :, 0:length] = x
        elif length > self.seq_length:
            new_x = x[:, :, 0:self.seq_length]
        else:
            new_x = x
        return new_x


# ---------------------------------------------------------------------------
# Signal augmentations (from GEM open_clip/augmentations/)
# ---------------------------------------------------------------------------

class BaselineWander(nn.Module):
    """Add low-frequency baseline wander noise.
    Aligned with GEM open_clip/augmentations/baseline_wander.py.
    """

    def __init__(
        self,
        max_amplitude=0.5,
        min_amplitude=0.0,
        p=1.0,
        max_freq=0.2,
        min_freq=0.01,
        k=3,
        fs=500,
        **kwargs,
    ):
        super().__init__()
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.k = k
        self.fs = fs
        self.p = p

    def forward(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0, 1):
            batch, csz, tsz = new_sample.shape
            amp_channel = np.random.normal(1, 0.5, size=(csz, 1))
            c = np.array([i for i in range(12)])
            amp_general = np.random.uniform(self.min_amplitude, self.max_amplitude, size=self.k)
            noise = np.zeros(shape=(1, tsz))
            for ki in range(self.k):
                noise += self._apply_baseline_wander(tsz) * amp_general[ki]
            noise = (noise * amp_channel).astype(np.float32)
            new_sample[:, c, :] = new_sample[:, c, :] + noise[c, :]
        return new_sample.float()

    def _apply_baseline_wander(self, tsz):
        f = np.random.uniform(self.min_freq, self.max_freq)
        t = np.linspace(0, tsz - 1, tsz)
        r = np.random.uniform(0, 2 * np.pi)
        noise = np.cos(2 * np.pi * f * (t / self.fs) + r)
        return noise


def _get_indices(prob: float, pop_size: int, scale_ratio: float = 0.1) -> List[int]:
    """Get a list of indices to be selected (from GEM RandomMasking)."""
    rng = np.random.default_rng()
    k = rng.normal(pop_size * prob, scale_ratio * pop_size)
    k = int(round(np.clip(k, 0, pop_size)))
    indices = rng.choice(list(range(pop_size)), k).tolist()
    return indices


class RandomMasking(nn.Module):
    """Randomly mask signal segments.
    Aligned with GEM open_clip/augmentations/RandomMasking.py.
    """

    def __init__(
        self,
        fs: int = 500,
        mask_value: float = 0.0,
        mask_width: Sequence[float] = (0.08, 0.18),
        prob: Sequence[float] = (0.3, 0.15),
        **kwargs,
    ):
        super().__init__()
        self.fs = fs
        self.prob = prob
        if isinstance(self.prob, Real):
            self.prob = np.array([self.prob, self.prob])
        else:
            self.prob = np.array(self.prob)
        assert (self.prob >= 0).all() and (self.prob <= 1).all()
        self.mask_value = mask_value
        self.mask_width = (np.array(mask_width) * self.fs).round().astype(int)

    def forward(self, sig: Tensor) -> Tensor:
        batch, lead, siglen = sig.shape
        sig_mask_prob = 0.5 / self.mask_width[1]
        sig_mask_scale_ratio = min(self.prob[1] / 4, 0.1) / self.mask_width[1]
        mask = torch.full_like(sig, 1, dtype=sig.dtype, device=sig.device)
        for batch_idx in _get_indices(prob=self.prob[0], pop_size=batch):
            indices = np.array(
                _get_indices(
                    prob=sig_mask_prob,
                    pop_size=siglen - self.mask_width[1],
                    scale_ratio=sig_mask_scale_ratio,
                )
            )
            indices += self.mask_width[1] // 2
            for j in indices:
                masked_radius = random.randint(self.mask_width[0], self.mask_width[1]) // 2
                mask[batch_idx, :, j - masked_radius: j + masked_radius] = self.mask_value
        return sig.mul_(mask)


class CutMix(nn.Module):
    """CutMix augmentation for 1-D signals.
    Aligned with GEM open_clip/augmentations/cutmix.py.
    """

    def __init__(
        self,
        fs: Optional[int] = None,
        num_mix: int = 1,
        alpha: float = 0.5,
        beta: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.fs = fs
        self.prob = 1.0
        self.num_mix = num_mix
        assert isinstance(self.num_mix, int) and self.num_mix > 0
        self.alpha = alpha
        self.beta = beta or self.alpha
        assert self.alpha > 0 and self.beta > 0

    def forward(self, sig: Tensor) -> Tensor:
        batch, lead, siglen = sig.shape
        rng = np.random.default_rng()
        for _ in range(self.num_mix):
            indices = np.arange(batch, dtype=int)
            ori = _get_indices(prob=self.prob, pop_size=batch)
            perm = deepcopy(ori)
            random_shuffle(perm)
            indices[ori] = perm
            indices = torch.from_numpy(indices).long()

            lam = torch.from_numpy(
                rng.beta(self.alpha, self.beta, size=batch)
            ).to(dtype=sig.dtype, device=sig.device)

            _lam = (lam.numpy() * siglen).astype(int)
            intervals = np.zeros((batch, 2), dtype=int)
            intervals[:, 0] = np.minimum(
                rng.integers(0, siglen, size=batch), siglen - _lam
            )
            intervals[:, 1] = intervals[:, 0] + _lam

            mask = torch.ones_like(sig)
            for i, (start, end) in enumerate(intervals):
                mask[i, :, start:end] = 0
            sig = sig * mask + sig[indices] * (1 - mask)
        return sig


# ---------------------------------------------------------------------------
# Compose helpers
# ---------------------------------------------------------------------------

class RandomApply(nn.Module):
    """Apply randomly a list of transformations with a given probability."""

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, x):
        if self.p < torch.rand(1):
            return x
        for t in self.transforms:
            x = t(x)
        return x


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.transform(x)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# ---------------------------------------------------------------------------
# Image augmentation (identical to GEM ImageAugmentation/augment.py)
# ---------------------------------------------------------------------------

class ImgAugTransform:
    """imgaug augmentation identical to GEM gem_generation/ecg-image-generator/ImageAugmentation/augment.py.

    Applies (training only):
      1. Affine rotation     : uniform [-rotate, +rotate] degrees
      2. AdditiveGaussianNoise: fixed scale
      3. Crop (percent)       : uniform [0, crop]
      4. ChangeColorTemperature: fixed kelvin value
    """

    def __init__(self, rotate: int = 25, noise: int = 25,
                 crop: float = 0.01, temperature: int = 6500):
        import imgaug.augmenters as iaa
        self._seq = iaa.Sequential([
            iaa.Affine(rotate=rotate),
            iaa.AdditiveGaussianNoise(scale=(noise, noise)),
            iaa.Crop(percent=(0, crop)),
            iaa.ChangeColorTemperature(temperature),
        ])

    def __call__(self, img):
        """Accepts a PIL Image, returns a PIL Image."""
        import numpy as np
        arr = np.array(img)
        augmented = self._seq(images=[arr[:, :, :3]])[0]
        from PIL import Image as PILImage
        return PILImage.fromarray(augmented)


# ---------------------------------------------------------------------------
# Transform builder functions
# ---------------------------------------------------------------------------

def get_signal_transform(
    is_train: bool,
    seq_length: int = 5000,
    use_augmentation: bool = True,
    use_baseline_wander: bool = True,
    use_cutmix: bool = False,
    use_random_masking: bool = True,
):
    """Build the 1-D signal transform pipeline (mirrors GEM ecg_transform)."""
    normalize = Normalize()
    resize = Resize(seq_length)
    if is_train and use_augmentation:
        transforms = []
        if use_baseline_wander:
            transforms.append(RandomApply([BaselineWander(fs=500)], p=0.5))
        if use_cutmix:
            transforms.append(RandomApply([CutMix(fs=500)], p=0.5))
        if use_random_masking:
            transforms.append(RandomApply([RandomMasking(fs=500)], p=0.3))
        transforms.extend([normalize, resize])
        return Compose(transforms)
    return Compose([normalize, resize])


def get_image_transform(is_train: bool, size: int = IMAGE_SIZE):
    """Build the 2-D image transform pipeline (CLIP-style, aligned with GEM).

    Both train and val use the same CLIP standard preprocessing:
    resize shortest_edge -> center-crop -> normalize.
    """
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
    ])
