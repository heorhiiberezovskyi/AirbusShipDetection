import lightning.pytorch as pl
import torch

from torch import Tensor
from torch import nn
from torch import optim
from torchvision.ops import sigmoid_focal_loss


class AirbusShipDetectorTrainingWrapper(pl.LightningModule):
    def __init__(self, ships_segmentor: nn.Module):
        super().__init__()
        self.ships_segmentor = ships_segmentor

        self._alpha = 0.25
        self._gamma = 2

    def training_step(self, batch, batch_idx):
        images = batch['image']
        gt_masks = batch['mask'].float()

        pred_mask_logits = self.ships_segmentor(images)

        dice = self.dice_coefficient(torch.sigmoid(pred_mask_logits), gt_masks)
        self.log("train_dice_coefficient", dice)

        dice_loss = 1 - dice
        self.log("train_dice_loss", dice_loss)

        focal_loss = sigmoid_focal_loss(inputs=pred_mask_logits, targets=gt_masks,
                                        alpha=self._alpha, gamma=self._gamma,
                                        reduction='mean')
        self.log("train_focal_loss", focal_loss)

        loss = focal_loss + dice_loss
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
        gt_masks = batch['mask'].float()

        pred_mask_logits = self.ships_segmentor(images)

        dice = self.dice_coefficient(torch.sigmoid(pred_mask_logits), gt_masks)
        self.log("val_dice_coefficient", dice)

        dice_loss = 1 - dice
        self.log("val_dice_loss", dice_loss)

        focal_loss = sigmoid_focal_loss(inputs=pred_mask_logits, targets=gt_masks,
                                        alpha=self._alpha, gamma=self._gamma,
                                        reduction='mean')
        self.log("val_focal_loss", focal_loss)

        loss = focal_loss + dice_loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
