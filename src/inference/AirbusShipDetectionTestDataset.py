from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class AirbusShipDetectionTestDataset(Dataset):
    def __init__(self, images_dir: str, image_names: List[str]):
        self._images_dir = images_dir
        self._image_names = image_names

        self._image_hw = (768, 768)

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        image_name = self._image_names[index]

        image_path = os.path.join(self._images_dir, image_name)
        image = cv2.imread(image_path)

        assert image is not None, image_path
        assert image.shape[:2] == self._image_hw

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        sample = {'image': image_tensor, 'image_name': image_name}
        return sample
