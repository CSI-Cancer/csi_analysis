from loguru import logger

import numpy as np

import torch
from cellpose import models

from csi_analysis.pipelines.scan_pipeline import MaskType, TileSegmenter
from csi_images.csi_scans import Scan
from csi_images.csi_images import make_rgb


class CellposeSegmenter(TileSegmenter):
    MASK_TYPE = MaskType.EVENT

    def __init__(
        self,
        scan: Scan,
        model_path: str = None,
        use_gpu: bool = False,
        save: bool = False,
    ):
        self.scan = scan
        # Preset: RGBW of AF555, AF647, DAPI, AF488
        self.colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        channels = ["AF555", "AF647", "DAPI", "AF488"]
        self.frame_order = scan.get_channel_indices(channels)
        self.model_path = model_path
        if self.model_path is None:
            # Use the built-in model
            self.model_path = "cyto3"
        self.use_gpu = use_gpu
        self.save = save
        if self.use_gpu:
            # Check if GPU is available
            if not torch.cuda.is_available():
                logger.warning("GPU requested but not available; using CPU")
                self.model = models.CellposeModel(pretrained_model=self.model_path)
            else:
                self.model = models.CellposeModel(
                    pretrained_model=self.model_path,
                    device=torch.device("cuda"),
                )
        else:
            self.model = models.CellposeModel(pretrained_model=self.model_path)

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.model_path})"

    def segment(self, images: list[np.ndarray]) -> dict[MaskType, np.ndarray]:
        ordered_frames = [images[i] for i in self.frame_order]
        rgb_image = make_rgb(ordered_frames, self.colors)
        mask, _, _ = self.model.eval(rgb_image, diameter=15, channels=[0, 0])
        return {self.MASK_TYPE: mask}
