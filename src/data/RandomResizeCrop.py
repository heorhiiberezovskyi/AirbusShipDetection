from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray

from src.data.SampleTransform import SampleTransform


class RandomResizeCrop(SampleTransform):
    def __init__(self, max_resize_dim: int, crop: SampleTransform):
        self._max_resize_dim = max_resize_dim
        self._crop = crop

    def apply(self, image: ndarray, mask: ndarray) -> Tuple[ndarray, ndarray]:
        random_dim = np.random.randint(self._max_resize_dim, 768)
        result_image = cv2.resize(image, (random_dim, random_dim), interpolation=cv2.INTER_AREA)
        result_mask = cv2.resize(mask, (random_dim, random_dim), interpolation=cv2.INTER_NEAREST)
        result_image, result_mask = self._crop.apply(image=result_image, mask=result_mask)
        return result_image, result_mask
