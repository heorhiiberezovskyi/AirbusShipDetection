from typing import Dict, List

import pandas as pd
from matplotlib import pyplot as plt

from src.data.AirbusShipDetectionDataset import to_dict


def get_ships_counts(encoded_pixels: Dict[str, List[str]], image_ids: List[str]) -> Dict[int, int]:
    ships_counts = {}
    for image_id in image_ids:
        if image_id not in encoded_pixels.keys():
            count = 0
        else:
            count = len(encoded_pixels[image_id])
        if count not in ships_counts.keys():
            ships_counts[count] = 0
        ships_counts[count] += 1
    result = dict(sorted(ships_counts.items()))
    return result


if __name__ == '__main__':
    root = r'D:\Data\airbus-ship-detection\train_v2'
    file = r'D:\Data\airbus-ship-detection\train_ship_segmentations_v2.csv'
    table = pd.read_csv(file, sep=',')

    unique_image_ids = table['ImageId'].unique().tolist()
    encoded_pixels_dict = to_dict(table)

    ships_counts = get_ships_counts(encoded_pixels=encoded_pixels_dict, image_ids=unique_image_ids)
    print(ships_counts)

    plt.bar(ships_counts.keys(), ships_counts.values())
    plt.show()

    # for key, encoded_pixels in encoded_pixels_dict.items():
    #     image_path = os.path.join(root, key)
    #     print(image_path)
    #     image = cv2.imread(image_path)
    #     mask = MaskVisualizer().visualize(encoded_pixels)
    #     alpha = mask / 4
    #     alpha = np.expand_dims(alpha, axis=2)
    #     mask_image = mask * 255
    #     mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    #
    #     color = np.full((768, 768, 3), fill_value=(0, 0, 255), dtype=np.uint8)
    #     blended = color * alpha + image * (1 - alpha)
    #     blended = blended.astype(np.uint8)
    #
    #     image = cv2.resize(image, (512, 512), cv2.INTER_AREA)
    #     mask_image = cv2.resize(mask_image, (512, 512), cv2.INTER_AREA)
    #     blended = cv2.resize(blended, (512, 512), cv2.INTER_AREA)
    #     stack = np.hstack([image, mask_image, blended])
    #     cv2.imshow('stack', stack)
    #     cv2.waitKey(0)
