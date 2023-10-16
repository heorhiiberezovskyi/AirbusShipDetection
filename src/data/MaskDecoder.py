from typing import Tuple, List

import numpy as np
from numpy import ndarray


def rle_decode(mask_rle: str, hw: Tuple[int, int]) -> ndarray:
    """"
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(hw[0] * hw[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(hw).T  # Needed to align to RLE direction


class MaskDecoder:
    def __init__(self, image_hw: Tuple[int, int]):
        self._image_hw = image_hw

    def decode(self, encodings: List[str]) -> ndarray:
        result = np.zeros(self._image_hw, dtype=np.uint8)
        for e in encodings:
            mask = rle_decode(e, hw=self._image_hw)
            result |= mask
        return result
