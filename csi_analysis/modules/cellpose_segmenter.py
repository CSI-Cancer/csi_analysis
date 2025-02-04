import os
from typing import Sequence

import numpy as np

import torch
from cellpose import models

from csi_images.csi_scans import Scan
from csi_images.csi_images import make_rgb

from csi_analysis.pipelines.scan import MaskType, TileSegmenter


class CellposeSegmenter(TileSegmenter):
    mask_type = MaskType.EVENT

    def __init__(
        self,
        scan: Scan,
        model: str = "cyto3",
        device: torch.device = torch.device("cpu"),
        channels: Sequence[str | int] = ("AF555", "AF647", "DAPI", "AF488"),
        colors: Sequence[tuple[float, float, float]] = (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
        ),
    ):
        """
        Creates a CellposeSegmenter object with custom configuration for
        segmenting events in images.
        :param scan: the scan to segment
        :param model: path to the model to use or name of a built-in model
        :param device: device to run the model on
        :param channels: list of channel names or indices to use
        :param colors: list of RGB colors to use for each channel
        """
        self.scan = scan
        self.channels = channels
        self.colors = colors
        if len(self.channels) != len(self.colors):
            raise ValueError("channels and colors must have the same length")
        if isinstance(self.channels[0], str):
            # Convert channel string names to indices
            self.channels = scan.get_channel_indices(self.channels)

        # Load the model
        self.model_name = os.path.basename(model)
        self.model = models.CellposeModel(pretrained_model=model, device=device)

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.model_name}"

    def segment(
        self, images: list[np.ndarray], masks: dict[MaskType, np.ndarray]
    ) -> dict[MaskType, np.ndarray]:
        """

        :param images: list of images from one of the scan's tiles to segment
        :param masks: dictionary of masks to add to, or None if none exists
        :return:
        """
        # Cellpose requires an RGB image, create based on options
        ordered_frames = [images[i] for i in self.channels]
        rgb_image = make_rgb(ordered_frames, self.colors)
        # Segment on all channels ([0, 0] = grayscale, no nuclear channel)
        mask, _, _ = self.model.eval(rgb_image, diameter=15, channels=[0, 0])
        masks[self.mask_type] = mask
        return masks
