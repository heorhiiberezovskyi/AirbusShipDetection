import json
import os

from src.data.AirbusShipDetectionDataset import AirbusShipDetectionDataset


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


if __name__ == '__main__':
    split_and_save_dataset(r'D:\Data\airbus-ship-detection')
