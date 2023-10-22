from typing import Tuple

import numpy as np
from numpy import ndarray

from src.data.SampleTransform import SampleTransform


class ShipCenteredCrop(SampleTransform):
    def __init__(self, hw: Tuple[int, int], center_crop_random_shift: float):
        self._hw = hw
        self._center_crop_random_shift = center_crop_random_shift

    def _get_random_crop_x0y0x1y1(self, image: ndarray, mask: ndarray) -> Tuple[int, int, int, int]:
        img_hw = image.shape[:2]
        if not mask.any():
            x0 = np.random.randint(img_hw[1] - self._hw[1])
            y0 = np.random.randint(img_hw[0] - self._hw[0])
            x1 = x0 + self._hw[1]
            y1 = y0 + self._hw[0]
        else:
            # Centered crop
            center_x, center_y = self._get_random_non_zero_pixel_xy(mask)
            # Shift center crop

            random_shift_factor_x = np.random.uniform(-self._center_crop_random_shift, self._center_crop_random_shift)
            random_shift_factor_y = np.random.uniform(-self._center_crop_random_shift, self._center_crop_random_shift)
            shift_x = int(random_shift_factor_x * self._hw[1])
            shift_y = int(random_shift_factor_y * self._hw[0])

            center_x += shift_x
            center_y += shift_y

            x0 = max(center_x - self._hw[1] // 2, 0)
            y0 = max(center_y - self._hw[0] // 2, 0)

            x1 = min(x0 + self._hw[1], img_hw[1])
            y1 = min(y0 + self._hw[0], img_hw[0])

            x0 = x1 - self._hw[1]
            y0 = y1 - self._hw[0]
        return x0, y0, x1, y1

    @staticmethod
    def _get_random_non_zero_pixel_xy(mask: ndarray) -> Tuple[int, int]:
        non_zero_y, non_zero_x = mask.nonzero()
        idx = np.random.randint(len(non_zero_y))
        x, y = non_zero_x[idx], non_zero_y[idx]
        return x, y

    def apply(self, image: ndarray, mask: ndarray) -> Tuple[ndarray, ndarray]:
        if image.shape[:2] == self._hw:
            return image, mask
        x0, y0, x1, y1 = self._get_random_crop_x0y0x1y1(image, mask)
        img_crop = image[y0: y1, x0: x1]
        mask_crop = mask[y0: y1, x0: x1]
        return img_crop, mask_crop
