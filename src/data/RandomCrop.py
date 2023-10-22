from typing import Tuple

import numpy as np
from numpy import ndarray

from src.data.SampleTransform import SampleTransform


class RandomCrop(SampleTransform):
    def __init__(self, hw: Tuple[int, int]):
        self._hw = hw

    def apply(self, image: ndarray, mask: ndarray) -> Tuple[ndarray, ndarray]:
        img_hw = image.shape[:2]
        x0 = np.random.randint(img_hw[1] - self._hw[1])
        y0 = np.random.randint(img_hw[0] - self._hw[0])
        x1 = x0 + self._hw[1]
        y1 = y0 + self._hw[0]
        img_crop = image[y0: y1, x0: x1]
        mask_crop = mask[y0: y1, x0: x1]
        return img_crop, mask_crop
