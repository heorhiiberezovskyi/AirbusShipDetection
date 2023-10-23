from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import Dataset

from src.data.MaskDecoder import MaskDecoder
from src.data.SampleTransform import SampleTransform


class AirbusShipDetectionDataset(Dataset):
    def __init__(self, images_dir: str, image_names: List[str], ship_encodings: Dict[str, List[str]]):
        self._images_dir = images_dir

        self._image_names = image_names
        self._ships_encodings = ship_encodings

        self._transform: Optional[SampleTransform] = None

        self._image_names_with_ships = list(ship_encodings.keys())
        self._image_names_without_ships = list(set(image_names).difference(self._image_names_with_ships))
        assert self._image_names_with_ships

        print('With ships: %s' % len(self._image_names_with_ships))
        print('Without ships: %s' % len(self._image_names_without_ships))

        self._mask_decoder = MaskDecoder(image_hw=(768, 768))

        self._rotate_prob = 0.5
        self._flip_prob = 0.5

        self._image_hw = (768, 768)

    def set_sample_transform(self, transform: SampleTransform):
        assert self._transform is None
        self._transform = transform

    def __len__(self):
        return len(self._image_names)

    def _get_random_balanced_image_name_and_ship_encodings(self, index: int) -> Tuple[str, List[str]]:
        ship_encodings = []
        if index % 2 == 0:
            random_image_with_ships_idx = np.random.randint(len(self._image_names_with_ships))
            image_name = self._image_names_with_ships[random_image_with_ships_idx]
            ship_encodings = self._ships_encodings[image_name]
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

    def __getitem__(self, index):
        image_name, ship_encodings = self._get_random_balanced_image_name_and_ship_encodings(index)

        image_path = os.path.join(self._images_dir, image_name)
        image = cv2.imread(image_path)

        assert image is not None, image_path
        assert image.shape[:2] == self._image_hw

        mask = self._mask_decoder.decode(ship_encodings)
        assert mask.shape[:2] == self._image_hw

        if self._transform is not None:
            image, mask = self._transform.apply(image=image, mask=mask)

        image, mask = self._apply_augmentations(image=image, mask=mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        mask_tensor = torch.from_numpy(np.transpose(np.expand_dims(mask, 2), (2, 0, 1))).long()

        sample = {'image': image_tensor, 'mask': mask_tensor, 'image_name': image_name}
        return sample

    def get_state(self) -> dict:
        return {'image_names': self._image_names.copy(),
                'ship_encodings': self._ships_encodings.copy()}

    @classmethod
    def from_state(cls, state: dict, images_dir: str) -> AirbusShipDetectionDataset:
        image_names = state['image_names']
        ship_encodings = state['ship_encodings']
        return AirbusShipDetectionDataset(images_dir=images_dir,
                                          image_names=image_names,
                                          ship_encodings=ship_encodings)
