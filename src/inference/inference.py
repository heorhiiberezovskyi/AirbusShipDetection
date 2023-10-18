import os
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.inference.AirbusShipDetectionTestDataset import AirbusShipDetectionTestDataset
from src.model.Unet import Unet
from src.train.AirbusShipDetectorTrainingWrapper import AirbusShipDetectorTrainingWrapper

if __name__ == '__main__':
    checkpoint = r'C:\Users\gosha\PycharmProjects\AirbusShipDetection\src\train\lightning_logs\version_2\checkpoints\epoch=6-step=37912.ckpt'

    state = torch.load(checkpoint)

    unet = Unet(init_channels=32, residual_block=True)
    unet.eval()
    unet.cuda()
    wrapper = AirbusShipDetectorTrainingWrapper(ships_segmentor=unet)

    wrapper.load_state_dict(state['state_dict'])

    test_imgs_dir = r'D:\Data\airbus-ship-detection\test_v2'
    predictions_save_dir = r'D:\Data\airbus-ship-detection\predictions_22'
    os.makedirs(predictions_save_dir, exist_ok=True)

    dataset = AirbusShipDetectionTestDataset(images_dir=test_imgs_dir, image_names=os.listdir(test_imgs_dir))

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=6,
                        pin_memory=True, persistent_workers=True)

    save_pool = ThreadPool(4)

    for batch in iter(loader):
        image = batch['image']
        image = image.cuda()
        image_name = batch['image_name'][0]

        with torch.no_grad():
            predicted_mask = unet(image)[0][0]

        predicted_mask = predicted_mask.detach().cpu().numpy()

        predicted_mask = predicted_mask * 255
        predicted_mask = predicted_mask.astype(np.uint8)
        save_path = os.path.join(predictions_save_dir, image_name)

        save_pool.apply_async(cv2.imwrite, (save_path, predicted_mask))
