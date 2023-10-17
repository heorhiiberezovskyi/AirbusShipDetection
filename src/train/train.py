import os

import lightning.pytorch as pl
import torch
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

from src.data.AirbusShipDetectionDataset import AirbusShipDetectionDataset
from src.model.Unet import Unet


class AirbusShipDetectorTrainingWrapper(pl.LightningModule):
    def __init__(self, ships_segmentor: nn.Module):
        super().__init__()
        self.ships_segmentor = ships_segmentor

        self.loss = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        images = batch['image']
        gt_masks = batch['mask'].float()

        pred_mask = self.ships_segmentor(images)

        dice = self.dice_coefficient(pred_mask, gt_masks)
        self.log("train_dice_coefficient", dice)

        loss = self.loss(pred_mask, gt_masks)
        self.log("train_loss", loss)
        return loss

    @staticmethod
    def dice_coefficient(y_pred: Tensor, y_true: Tensor):
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice = (2.0 * intersection) / (union + 1e-8)
        return dice

    def validation_step(self, batch, batch_idx):
        images = batch['image']

        pred_mask = self.ships_segmentor(images)

        dice = self.dice_coefficient(pred_mask, batch['mask'].float())
        self.log("val_dice_coefficient", dice)
        return dice

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    unet = Unet(init_channels=32, residual_block=True)
    detector = AirbusShipDetectorTrainingWrapper(unet)

    dataset = AirbusShipDetectionDataset.initialize(
        images_dir=r'D:\Data\airbus-ship-detection\train_v2',
        annotations_file=r'D:\Data\airbus-ship-detection\train_ship_segmentations_v2.csv')
    train_dataset, val_dataset = dataset.split_train_val(train_percent=0.9)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=True, num_workers=6, pin_memory=True)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(max_epochs=10, logger=[CSVLogger(os.getcwd()), TensorBoardLogger(os.getcwd())],
                         enable_progress_bar=True, enable_checkpointing=True,
                         val_check_interval=0.25, log_every_n_steps=10)
    trainer.fit(model=detector, train_dataloaders=train_loader, val_dataloaders=val_loader)
