import argparse
import json
import os
from multiprocessing.pool import ThreadPool
from typing import Dict

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.ResizeImageOnly import ResizeImageOnly
from src.inference.AirbusShipDetectionTestDataset import AirbusShipDetectionTestDataset
from src.model.Unet import Unet
from src.train.AirbusShipDetectorTrainingWrapper import AirbusShipDetectorTrainingWrapper


def init_from_meta_info(images_dir: str, annotations_json: str) -> AirbusShipDetectionTestDataset:
    with open(annotations_json, 'r') as file:
        state = json.load(file)
    return AirbusShipDetectionTestDataset.from_state(state=state, images_dir=images_dir)


def load_and_prepare_unet(checkpoint: str) -> Unet:
    state = torch.load(checkpoint)

    unet = Unet(init_channels=32, residual_block=True)
    unet.eval()
    unet.cuda()
    wrapper = AirbusShipDetectorTrainingWrapper(ships_segmentor=unet)

    wrapper.load_state_dict(state['state_dict'])
    return unet


def calculate_counts(pred_mask: Tensor, gt_mask: Tensor, eps: float = 1e-8):
    tp = torch.sum(pred_mask * gt_mask)  # TP
    fp = torch.sum(pred_mask * (1 - gt_mask))  # FP
    fn = torch.sum((1 - pred_mask) * gt_mask)  # FN
    tn = torch.sum((1 - pred_mask) * (1 - gt_mask))  # TN
    return {'tp': tp.item(),
            'fp': fp.item(),
            'fn': fn.item(),
            'tn': tn.item()}


def sum_counts(counts: Dict[str, Dict[str, int]]) -> Dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    for c in counts.values():
        total_tp += c['tp']
        total_fp += c['fp']
        total_fn += c['fn']
        total_tn += c['tn']
    return {'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'tn': total_tn}


def calculate_metrics(counts: dict, eps: float = 1e-8) -> dict:
    tp = counts['tp']
    fp = counts['fp']
    fn = counts['fn']
    tn = counts['tn']

    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return {'pixel_acc': pixel_acc,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'specificity': specificity}


def main(args):
    predictions_save_dir = args.save_dir
    os.makedirs(predictions_save_dir, exist_ok=True)

    val_dataset = init_from_meta_info(images_dir=args.images_dir, annotations_json=args.annotations_json)

    predictions_upsample_factor = None
    if args.input_resolution is not None:
        val_dataset.set_sample_transform(ResizeImageOnly((args.input_resolution, args.input_resolution)))
        predictions_upsample_factor = 768 / args.input_resolution
        print('Predicting in different resolution: %s' % args.input_resolution)
    else:
        print('Predicting in original resolution...')

    model = load_and_prepare_unet(args.ckpt)

    data_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=6,
                             pin_memory=True, persistent_workers=True)

    save_pool = ThreadPool(4)
    all_counts = {}
    idx = 0
    total = len(val_dataset)
    for batch in iter(data_loader):
        image = batch['image'].cuda()
        gt_mask = batch['mask'][0][0].cuda()

        with torch.no_grad():
            pred_mask_logits = model(image)

            if predictions_upsample_factor is not None:
                pred_mask_logits = F.interpolate(pred_mask_logits, scale_factor=predictions_upsample_factor,
                                                 mode='bilinear')

            predicted_mask = torch.sigmoid(pred_mask_logits)[0][0]

            assert tuple(predicted_mask.size()) == (768, 768)
            counts = calculate_counts(pred_mask=predicted_mask, gt_mask=gt_mask)

        image_name = batch['image_name'][0]
        all_counts[image_name] = counts

        predicted_mask = predicted_mask.detach().cpu().numpy()

        predicted_mask = predicted_mask * 255
        predicted_mask = predicted_mask.astype(np.uint8)

        save_path = os.path.join(predictions_save_dir, image_name)

        save_pool.apply_async(cv2.imwrite, (save_path, predicted_mask))
        if idx % 10 == 0:
            print('%s out of %s' % (idx, total))
        idx += 1

    save_pool.close()
    save_pool.join()

    with open(os.path.join(predictions_save_dir, 'all_counts.json'), 'w') as file:
        json.dump(all_counts, file)

    total_counts = sum_counts(counts=all_counts)
    with open(os.path.join(predictions_save_dir, 'total_counts.json'), 'w') as file:
        json.dump(total_counts, file)

    metrics = calculate_metrics(counts=total_counts)
    with open(os.path.join(predictions_save_dir, 'metrics_avg.json'), 'w') as file:
        json.dump(metrics, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval model and calculate metrics on validation set.')
    parser.add_argument("--input_resolution", type=str, default=None,
                        help="Resolution to run inference on. If None, use original images resolution")
    parser.add_argument("--ckpt", type=str, help="path to snapshot in .ckpt format")
    parser.add_argument("--save_dir", type=str, help="Directory to dump predictions and metrics.")
    parser.add_argument("--images_dir", type=str, help="Directory with images to evaluate on.")
    parser.add_argument("--annotations_json", type=str, help="Path to JSON file with ship segmentations annotations")
    _args = parser.parse_args()
    main(_args)
