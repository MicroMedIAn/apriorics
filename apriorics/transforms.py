import random
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from albumentations import CropNonEmptyMaskIfExists
from albumentations.core.transforms_interface import DualTransform
from pathaia.util.types import NDGrayImage, NDImage
from skimage.morphology import label, remove_small_holes, remove_small_objects
from torchvision.transforms.functional import to_tensor


class ToSingleChannelMask(DualTransform):
    """
    Transforms that takes a grayscale masks with rgb or rgba channels and transform them
    into a single channel image

    Target : mask, masks
    Type : any
    """

    def __init__(self, trailing_channels: bool = True):
        super().__init__(True, 1)
        self.trailing_channels = trailing_channels

    def apply(self, img: NDImage, **params) -> NDImage:
        return img

    def apply_to_mask(self, img: NDImage, **params) -> NDGrayImage:
        if self.trailing_channels:
            return img[:, :, 0]
        else:
            return img[0]


class DropAlphaChannel(DualTransform):
    """
    Transform that takes rgba images and mask and that removes the alpha channel

    Target : image, mask, masks
    Type : any
    """

    def __init__(self, trailing_channels: bool = True):
        super().__init__(True, 1)
        self.trailing_channels = trailing_channels

    def apply(self, img: NDImage, **params) -> NDImage:
        if self.trailing_channels:
            assert img.shape[2] == 4
            return img[:, :, :-1]
        else:
            assert img.shape[0] == 4
            return img[:-1]


class ToTensor(DualTransform):
    def __init__(
        self, transpose_mask: bool = False, always_apply: bool = True, p: float = 1
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> Dict[str, Callable[[NDImage], torch.Tensor]]:
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img: NDImage, **params) -> torch.Tensor:
        return to_tensor(img)

    def apply_to_mask(self, mask: NDImage, **params) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("transpose_mask",)


class RandomCropAroundMaskIfExists(CropNonEmptyMaskIfExists):
    """Crop area with mask if mask is non-empty, else make random crop. Cropped area
    will always be centered around a non empty area with a random offset.
    Args:
        height: vertical size of crop in pixels
        width: horizontal size of crop in pixels
        ignore_values: values to ignore in mask, `0` values are always ignored
            (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
        ignore_channels: channels to ignore in mask
            (e.g. if background is a first channel set `ignore_channels=[0]` to ignore)
        p: probability of applying the transform. Default: 1.0.
    """

    def __init__(
        self,
        height: int,
        width: int,
        ignore_values: Optional[Sequence[int]] = None,
        ignore_channels: Optional[Sequence[int]] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(RandomCropAroundMaskIfExists, self).__init__(
            height,
            width,
            ignore_values=ignore_values,
            ignore_channels=ignore_channels,
            always_apply=always_apply,
            p=p,
        )
        self.height = height
        self.width = width

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(masks[0])
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            labels, n = label(mask, return_num=True)
            idx = random.randint(1, n)
            mask = labels == idx
            non_zero_yx = np.argwhere(mask)
            ymin, xmin = non_zero_yx.min(0)
            ymax, xmax = non_zero_yx.max(0)
            x_min = random.randint(
                max(0, xmax - self.width), min(xmin, mask_width - self.width)
            )
            y_min = random.randint(
                max(0, ymax - self.width), min(ymin, mask_width - self.width)
            )
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params


class FixedCropAroundMaskIfExists(CropNonEmptyMaskIfExists):
    """Crop area with mask if mask is non-empty, else make center crop. Cropped area
    will always be centered around a non empty area in a fully deterministic way.
    Args:
        height: vertical size of crop in pixels
        width: horizontal size of crop in pixels
        ignore_values: values to ignore in mask, `0` values are always ignored
            (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
        ignore_channels: channels to ignore in mask
            (e.g. if background is a first channel set `ignore_channels=[0]` to ignore)
        p: probability of applying the transform. Default: 1.0.
    """

    def __init__(
        self,
        height: int,
        width: int,
        ignore_values: Optional[Sequence[int]] = None,
        ignore_channels: Optional[Sequence[int]] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(FixedCropAroundMaskIfExists, self).__init__(
            height,
            width,
            ignore_values=ignore_values,
            ignore_channels=ignore_channels,
            always_apply=always_apply,
            p=p,
        )
        self.height = height
        self.width = width

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(masks[0])
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for FixedCropAroundMaskIfExists")

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            shape = np.array([self.height, self.width], dtype=np.int64)
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            labels, n = label(mask, return_num=True)
            mask = np.zeros_like(mask)
            for i in range(1, n + 1):
                if (labels == i).sum() > mask.sum():
                    mask = labels == i
            non_zero_yx = np.argwhere(mask)
            center = non_zero_yx.mean(axis=0, dtype=np.int64)
            y_min, x_min = np.maximum(center - shape, 0)
            y_min = min(y_min, mask_height - self.height)
            x_min = min(x_min, mask_width - self.width)
        else:
            y_min = (mask_height - self.height) // 2
            x_min = (mask_width - self.width) // 2

        x_max = x_min + self.width
        y_max = y_min + self.height

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params


class CorrectCompression(DualTransform):
    def __init__(
        self,
        min_size: int = 10,
        area_threshold: int = 10,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.min_size = min_size
        self.area_threshold = area_threshold

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        if mask.ndim == 3:
            new_mask = mask[:, :, 0] > 0
        else:
            new_mask = mask > 0
        new_mask = remove_small_objects(
            remove_small_holes(new_mask, area_threshold=self.area_threshold),
            min_size=self.min_size,
        )
        new_mask = new_mask.astype(mask.dtype)
        if mask.dtype == np.uint8:
            new_mask *= 255
        if mask.ndim == 3:
            new_mask = np.repeat(new_mask[:, :, None], mask.shape[-1], axis=-1)
        return new_mask

    def get_transform_init_args_names(self):
        return ("min_size", "area_threshold")
