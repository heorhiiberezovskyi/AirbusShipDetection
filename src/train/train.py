import os

import lightning.pytorch as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from src.data.AirbusShipDetectionDataset import AirbusShipDetectionDataset
from src.model.Unet import Unet
from src.train.AirbusShipDetectorTrainingWrapper import AirbusShipDetectorTrainingWrapper

if __name__ == '__main__':
    unet = Unet(init_channels=32, residual_block=True)
    detector = AirbusShipDetectorTrainingWrapper(unet)

    dataset = AirbusShipDetectionDataset.initialize(
        images_dir=r'D:\Data\airbus-ship-detection\train_v2',
        annotations_file=r'D:\Data\airbus-ship-detection\train_ship_segmentations_v2.csv',
        crop_hw=(256, 256),
        test=False)
    train_dataset, val_dataset = dataset.split_train_val(train_percent=0.9)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=True, num_workers=6, pin_memory=True,
                            persistent_workers=True)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(max_epochs=10, logger=[CSVLogger(os.getcwd()), TensorBoardLogger(os.getcwd())],
                         enable_progress_bar=True, enable_checkpointing=True,
                         val_check_interval=0.25, log_every_n_steps=10)
    trainer.fit(model=detector, train_dataloaders=train_loader, val_dataloaders=val_loader)
