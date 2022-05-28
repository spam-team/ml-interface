import numpy as np
from typing import Tuple

from segmentation import segment_walruses
from utils.cluster_counter import SegmentCounter


def get_walrus_count(img: np.array) -> Tuple[np.array, np.array, int]:
    counter = SegmentCounter()

    mask = segment_walruses(img)
    centroids = counter.get_centroids(mask)
    count = len(centroids)

    return mask, centroids, count
