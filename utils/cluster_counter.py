import math

import faiss
import numpy as np
from sklearn.cluster import DBSCAN

from typing import Tuple


class SegmentCounter:
    DEFAULT_SIZE = (1090, 920)

    def _get_mask_points(self, mask: np.array) -> np.array:
        points = np.array([
            (i, j)
            for i in range(mask.shape[0])
            for j in range(mask.shape[1])
            if mask[i, j] != 0
        ])

        return points

    def _get_count_by_area(self, points: np.array) -> int:
        dbscan = DBSCAN(eps=1)
        dbscan.fit(points)

        counts = np.unique(dbscan.labels_, return_counts=True)[1]
        m_count = np.median(counts)
        count = math.ceil(len(points) / m_count)

        return count

    def get_animal_count(self, segment_mask: np.array) -> int:
        points = self._get_mask_points(segment_mask)
        count = self._get_count_by_area(points)
        return count

    def get_centroids(self, segment_mask: np.array) -> Tuple[np.array, int]:
        points = self._get_mask_points(segment_mask)
        animal_count = self.get_animal_count(points)

        niter = 3
        verbose = False
        d = points.shape[1]
        kmeans = faiss.Kmeans(d, animal_count, niter=niter, verbose=verbose, gpu=False)
        kmeans.train(points.astype('float32'))

        return kmeans.centroids
