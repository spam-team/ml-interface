import numpy as np
from typing import Tuple

from segmentation import segment_walruses
from utils.cluster_counter import SegmentCounter


def get_walrus_count(img: np.array) -> Tuple[np.array, int]:
    counter = SegmentCounter()

    mask = segment_walruses(img)
    count = counter.get_animal_count(mask)

    return mask, count


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    img = cv2.imread('./examples/morz.jpeg')

    mask, count = get_walrus_count(img)
    plt.title(f'{count} моржей найдено')
    plt.imshow(mask)
    plt.show()
