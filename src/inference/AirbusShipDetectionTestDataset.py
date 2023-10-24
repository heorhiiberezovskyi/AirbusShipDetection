from __future__ import annotations

import os
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.MaskDecoder import MaskDecoder
from src.data.SampleTransform import SampleTransform


class AirbusShipDetectionTestDataset(Dataset):
    def __init__(self, images_dir: str, image_names: List[str], ship_encodings: Dict[str, List[str]]):
        self._images_dir = images_dir

        self._image_names = image_names
        self._ships_encodings = ship_encodings

        self._image_hw = (768, 768)

        self._transform: Optional[SampleTransform] = None

        self._image_names_with_ships = list(ship_encodings.keys())
        self._image_names_without_ships = list(set(image_names).difference(self._image_names_with_ships))
        assert self._image_names_with_ships

        print('With ships: %s' % len(self._image_names_with_ships))
        print('Without ships: %s' % len(self._image_names_without_ships))

        self._mask_decoder = MaskDecoder(image_hw=self._image_hw)

    def set_sample_transform(self, transform: SampleTransform):
        assert self._transform is None
        self._transform = transform

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        image_name = self._image_names[index]

        image_path = os.path.join(self._images_dir, image_name)
        image = cv2.imread(image_path)

        assert image is not None, image_path
        assert image.shape[:2] == self._image_hw

        ship_encodings = self._ships_encodings[image_name] if image_name in self._ships_encodings.keys() else []

        mask = self._mask_decoder.decode(ship_encodings)
        assert mask.shape[:2] == self._image_hw

        if self._transform is not None:
            image, mask = self._transform.apply(image=image, mask=mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        mask_tensor = torch.from_numpy(np.transpose(np.expand_dims(mask, 2), (2, 0, 1))).long()
        sample = {'image': image_tensor, 'image_name': image_name, 'mask': mask_tensor}
        return sample

    @classmethod
    def from_state(cls, state: dict, images_dir: str) -> AirbusShipDetectionTestDataset:
        image_names = state['image_names']
        ship_encodings = state['ship_encodings']
        return AirbusShipDetectionTestDataset(images_dir=images_dir,
                                              image_names=image_names,
                                              ship_encodings=ship_encodings)
