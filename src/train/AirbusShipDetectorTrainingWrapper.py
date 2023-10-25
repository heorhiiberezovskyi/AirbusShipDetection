import lightning.pytorch as pl
import torch
from torch import Tensor
from torch import nn
from torch import optim
from torch.nn import functional as F
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

        bce_loss = F.binary_cross_entropy_with_logits(input=pred_mask_logits,
                                                      target=gt_masks,
                                                      reduction='mean')
        self.log('train_bce_loss', bce_loss)

        focal_loss = sigmoid_focal_loss(inputs=pred_mask_logits, targets=gt_masks,
                                        alpha=self._alpha, gamma=self._gamma,
                                        reduction='mean')
        self.log("train_focal_loss", focal_loss)

        loss = focal_loss + dice_loss + bce_loss
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
        assert tuple(gt_masks.size()[2:]) == (768, 768)

        pred_mask_logits = self.ships_segmentor(images)

        # Calculate validation metrics on original size images.
        if tuple(pred_mask_logits.size()[2:]) != (768, 768):
            scale_factor = 768 / pred_mask_logits.size()[2]
            pred_mask_logits = F.interpolate(pred_mask_logits, scale_factor=scale_factor, mode='bilinear')
            assert tuple(pred_mask_logits.size()[2:]) == (768, 768)

        dice = self.dice_coefficient(torch.sigmoid(pred_mask_logits), gt_masks)
        self.log("val_dice_coefficient", dice)

        dice_loss = 1 - dice
        self.log("val_dice_loss", dice_loss)

        bce_loss = F.binary_cross_entropy_with_logits(input=pred_mask_logits,
                                                      target=gt_masks,
                                                      reduction='mean')
        self.log('val_bce_loss', bce_loss)

        focal_loss = sigmoid_focal_loss(inputs=pred_mask_logits, targets=gt_masks,
                                        alpha=self._alpha, gamma=self._gamma,
                                        reduction='mean')
        self.log("val_focal_loss", focal_loss)

        loss = focal_loss + dice_loss + bce_loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
