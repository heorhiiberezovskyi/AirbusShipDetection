import json
import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from src.data.AirbusShipDetectionDataset import AirbusShipDetectionDataset
from src.data.ResizeImageMask import ResizeImageMask
from src.data.ResizeImageOnly import ResizeImageOnly
from src.model.Unet import Unet
from src.train.AirbusShipDetectorTrainingWrapper import AirbusShipDetectorTrainingWrapper


def init_from_meta_info(images_dir: str, annotations_json: str, balanced_sampling: bool) -> AirbusShipDetectionDataset:
    with open(annotations_json, 'r') as file:
        state = json.load(file)
    return AirbusShipDetectionDataset.from_state(state=state,
                                                 images_dir=images_dir,
                                                 balanced_sampling=balanced_sampling)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32 - 1)
    print('Worker ' + str(worker_id) + ' seed set to ' + str(seed))
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # sampling: balanced
    # 4_5 - focal + dice
    # 6_7 - + bce
    # 8_9 - resize to 256
    # 10_11 - random resize from 768 to 384 and ship centered crop
    # 12_13 - random resize from 768 to 384 and random crop
    # 14_15 - random resize from 768 to 256 and ship centered crop
    # 20_21 - resize to 256
    # 22_23 - resize to 256, random sampling

    unet = Unet(init_channels=32, residual_block=True)
    detector = AirbusShipDetectorTrainingWrapper(unet)

    images_root = r'D:\Data\airbus-ship-detection\train_v2'
    train_dataset = init_from_meta_info(images_dir=images_root,
                                        annotations_json=r'D:\Data\airbus-ship-detection\train.json',
                                        balanced_sampling=False)
    val_dataset = init_from_meta_info(images_dir=images_root,
                                      annotations_json=r'D:\Data\airbus-ship-detection\val.json',
                                      balanced_sampling=True)

    # Set centered crop transform to train dataset, perform validation in original size.
    # crop = ShipCenteredCrop(hw=(256, 256), center_crop_random_shift=0.3)
    # crop = RandomCrop(hw=(256, 256))
    # train_dataset.set_sample_transform(RandomResizeCrop(max_resize_dim=256, crop=crop))
    train_dataset.set_sample_transform(ResizeImageMask((256, 256)))
    val_dataset.set_sample_transform(ResizeImageOnly((256, 256)))

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)

    trainer = pl.Trainer(max_epochs=5, logger=[CSVLogger(os.getcwd()), TensorBoardLogger(os.getcwd())],
                         enable_progress_bar=True, enable_checkpointing=True,
                         val_check_interval=0.25, log_every_n_steps=10,
                         limit_val_batches=200)
    trainer.fit(model=detector, train_dataloaders=train_loader, val_dataloaders=val_loader)
