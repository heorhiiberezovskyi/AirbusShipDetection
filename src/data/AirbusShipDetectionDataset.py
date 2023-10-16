import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from torch.utils.data import Dataset

from src.data.MaskDecoder import MaskDecoder


def to_dict(table: DataFrame) -> Dict[str, List[str]]:
    state_dict = {}
    not_nan = table['EncodedPixels'].notna()
    for index, row in table[not_nan].iterrows():
        image_id = row['ImageId']
        if image_id not in state_dict.keys():
            state_dict[image_id] = []
        na = row.isna()
        if not na['EncodedPixels']:
            state_dict[image_id].append(row['EncodedPixels'])
    return state_dict


class AirbusShipDetectionDataset(Dataset):
    def __init__(self, images_dir: str, annotations_file: str):
        self._images_dir = images_dir

        table = pd.read_csv(annotations_file, sep=',')

        self._all_image_names = table['ImageId'].unique().tolist()
        self._ships_encodings_dict = to_dict(table)

        self._image_names_with_ships = list(self._ships_encodings_dict.keys())
        self._image_names_without_ships = list(set(self._all_image_names).difference(self._image_names_with_ships))

        self._mask_decoder = MaskDecoder(image_hw=(768, 768))

        self._rotate_prob = 0.5
        self._flip_prob = 0.5

        self._crop_hw = (256, 256)
        self._image_hw = (768, 768)

    def __len__(self):
        return len(self._all_image_names)

    def _get_random_balanced_image_name_and_ship_encodings(self, index: int) -> Tuple[str, List[str]]:
        ship_encodings = []
        if index % 2 == 0:
            random_image_with_ships_idx = np.random.randint(len(self._image_names_with_ships))
            image_name = self._image_names_with_ships[random_image_with_ships_idx]
            ship_encodings = self._ships_encodings_dict[image_name]
        else:
            image_name_without_ship_idx = np.random.randint(len(self._image_names_without_ships))
            image_name = self._image_names_without_ships[image_name_without_ship_idx]
        return image_name, ship_encodings

    def _apply_augmentations(self, image: ndarray, mask: ndarray) -> Tuple[ndarray, ndarray]:
        do_rotate = np.random.rand() < self._rotate_prob
        if do_rotate:
            directions = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
            random_direction_idx = np.random.randint(len(directions))
            random_direction = directions[random_direction_idx]
            image = cv2.rotate(image, rotateCode=random_direction)
            mask = cv2.rotate(mask, rotateCode=random_direction)

        do_flip = np.random.rand() < self._flip_prob
        if do_flip:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask

    @staticmethod
    def _get_random_non_zero_pixel_xy(mask: ndarray) -> Tuple[int, int]:
        non_zero_y, non_zero_x = mask.nonzero()
        idx = np.random.randint(len(non_zero_y))
        x, y = non_zero_x[idx], non_zero_y[idx]
        return x, y

    def _get_random_crop_x0y0x1y1(self, image: ndarray, mask: ndarray) -> Tuple[int, int, int, int]:
        img_hw = image.shape[:2]
        if not mask.any():
            x0 = np.random.randint(img_hw[1] - self._crop_hw[1])
            y0 = np.random.randint(img_hw[0] - self._crop_hw[0])
            x1 = x0 + self._crop_hw[1]
            y1 = y0 + self._crop_hw[0]
        else:
            # Centered crop
            center_x, center_y = self._get_random_non_zero_pixel_xy(mask)
            x0 = max(center_x - self._crop_hw[1] // 2, 0)
            y0 = max(center_y - self._crop_hw[0] // 2, 0)

            x1 = min(x0 + self._crop_hw[1], img_hw[1])
            y1 = min(y0 + self._crop_hw[0], img_hw[0])

            x0 = x1 - self._crop_hw[1]
            y0 = y1 - self._crop_hw[0]
        return x0, y0, x1, y1

    def _random_crop(self, image: ndarray, mask: ndarray) -> Tuple[ndarray, ndarray]:
        x0, y0, x1, y1 = self._get_random_crop_x0y0x1y1(image, mask)
        img_crop = image[y0: y1, x0: x1]
        mask_crop = mask[y0: y1, x0: x1]
        return img_crop, mask_crop

    def __getitem__(self, index):
        image_name, ship_encodings = self._get_random_balanced_image_name_and_ship_encodings(index)

        image_path = os.path.join(self._images_dir, image_name)
        image = cv2.imread(image_path)

        assert image is not None, image_path
        assert image.shape[:2] == self._image_hw

        mask = self._mask_decoder.decode(ship_encodings)
        assert mask.shape[:2] == self._image_hw

        image, mask = self._random_crop(image=image, mask=mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, mask = self._apply_augmentations(image=image, mask=mask)
        assert image.shape[:2] == self._crop_hw
        assert mask.shape[:2] == self._crop_hw

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        mask_tensor = torch.from_numpy(np.transpose(np.expand_dims(mask, 2), (2, 0, 1))).long()

        sample = {'image': image_tensor, 'mask': mask_tensor}
        return sample
