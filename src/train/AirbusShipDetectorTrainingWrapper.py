import lightning.pytorch as pl
import torch
from torch import Tensor
from torch import nn
from torch import optim


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
