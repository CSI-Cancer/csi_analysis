"""
Module for tophat preprocessing of frame images.
"""

import numpy as np
import cv2

from csi_images.csi_scans import Scan
from ..pipelines.scan_pipeline import TilePreprocessor


class TophatPreprocessor(TilePreprocessor):
    def __init__(
        self,
        scan: Scan,
        channels: list[int | str],
        tophat_size: int = 0,
    ):
        """
        Preprocess frame images with tophat filtering.
        :param scan:
        :param channels:
        :param tophat_size:
        """
        self.scan = scan
        self.channels = channels
        if isinstance(channels[0], str):
            self.channels = scan.get_channel_indices(channels)
        self.tophat_size = tophat_size

    def preprocess(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if self.tophat_size == 0:
            return images

        tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.tophat_size, self.tophat_size)
        )

        for i in self.channels:
            images[i] = cv2.morphologyEx(images[i], cv2.MORPH_TOPHAT, tophat_kernel)

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.tophat_size})"
