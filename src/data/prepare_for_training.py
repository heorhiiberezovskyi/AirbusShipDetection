import json
import os
from typing import Dict, List

import pandas as pd
from pandas import DataFrame

from src.data.AirbusShipDetectionDataset import AirbusShipDetectionDataset


def to_dict(table: DataFrame) -> Dict[str, List[str]]:
    state_dict = {}
    not_nan = table['EncodedPixels'].notna()
    for index, row in table[not_nan].iterrows():
        image_id = row['ImageId']
        if image_id not in state_dict.keys():
            state_dict[image_id] = []
        na = row.isna()
        if not na['EncodedPixels']:
            state_dict[image_id].append(row['EncodedPixels'])
    return state_dict


def split_and_save_dataset(data_root: str):
    images_dir = os.path.join(data_root, 'train_v2')
    annotations_file = os.path.join(data_root, 'train_ship_segmentations_v2.csv')

    table = pd.read_csv(annotations_file, sep=',')

    image_names = table['ImageId'].unique().tolist()
    ships_encodings = to_dict(table)
    dataset = AirbusShipDetectionDataset(images_dir=images_dir,
                                         image_names=image_names,
                                         ship_encodings=ships_encodings)

    train_dataset, val_dataset = dataset.split_train_val(train_percent=0.9)

    train_file = os.path.join(data_root, 'train.json')
    val_file = os.path.join(data_root, 'val.json')

    with open(train_file, 'w') as file:
        json.dump(train_dataset.get_state(), file)

    with open(val_file, 'w') as file:
        json.dump(val_dataset.get_state(), file)


if __name__ == '__main__':
    split_and_save_dataset(r'D:\Data\airbus-ship-detection')
