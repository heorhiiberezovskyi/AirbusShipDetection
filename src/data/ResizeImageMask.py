from typing import Tuple

import cv2
from numpy import ndarray

from src.data.SampleTransform import SampleTransform


class ResizeImageMask(SampleTransform):
    def __init__(self, hw: Tuple[int, int]):
        self._hw = hw

    def apply(self, image: ndarray, mask: ndarray) -> Tuple[ndarray, ndarray]:
        result_image = cv2.resize(image, (self._hw[1], self._hw[0]), interpolation=cv2.INTER_AREA)
        result_mask = cv2.resize(mask, (self._hw[1], self._hw[0]), interpolation=cv2.INTER_NEAREST)
        return result_image, result_mask
