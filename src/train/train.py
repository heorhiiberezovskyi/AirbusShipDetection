import json
import os
import random
from typing import Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from src.data.AirbusShipDetectionDataset import AirbusShipDetectionDataset
from src.model.Unet import Unet
from src.train.AirbusShipDetectorTrainingWrapper import AirbusShipDetectorTrainingWrapper


def init_train_val_from_meta_info(data_root: str) -> Tuple[AirbusShipDetectionDataset, AirbusShipDetectionDataset]:
    images_dir = os.path.join(data_root, 'train_v2')
    train_file = os.path.join(data_root, 'train.json')
    with open(train_file, 'r') as file:
        train_state = json.load(file)

    val_file = os.path.join(data_root, 'val.json')
    with open(val_file, 'r') as file:
        val_state = json.load(file)

    train_dataset = AirbusShipDetectionDataset.from_state(state=train_state,
                                                          images_dir=images_dir)
    val_dataset = AirbusShipDetectionDataset.from_state(state=val_state,
                                                        images_dir=images_dir)

    return train_dataset, val_dataset


def split_and_save_dataset(data_root: str):
    images_dir = os.path.join(data_root, 'train_v2')
    annotations_file = os.path.join(data_root, 'train_ship_segmentations_v2.csv')
    dataset = AirbusShipDetectionDataset.initialize(images_dir=images_dir,
                                                    annotations_file=annotations_file)
    train_dataset, val_dataset = dataset.split_train_val(train_percent=0.9)

    train_file = os.path.join(data_root, 'train.json')
    val_file = os.path.join(data_root, 'val.json')

    with open(train_file, 'w') as file:
        json.dump(train_dataset.get_state(), file)

    with open(val_file, 'w') as file:
        json.dump(val_dataset.get_state(), file)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32 - 1)
    print('Worker ' + str(worker_id) + ' seed set to ' + str(seed))
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # split_and_save_dataset(r'D:\Data\airbus-ship-detection')

    unet = Unet(init_channels=32, residual_block=True, inference=False)
    detector = AirbusShipDetectorTrainingWrapper(unet)

    train_dataset, val_dataset = init_train_val_from_meta_info(r'D:\Data\airbus-ship-detection')
    train_dataset.set_crop_hw((256, 256))

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False, num_workers=6, pin_memory=True,
                            persistent_workers=True)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(max_epochs=5, logger=[CSVLogger(os.getcwd()), TensorBoardLogger(os.getcwd())],
                         enable_progress_bar=True, enable_checkpointing=True,
                         val_check_interval=0.25, log_every_n_steps=10)
    trainer.fit(model=detector, train_dataloaders=train_loader, val_dataloaders=val_loader)
