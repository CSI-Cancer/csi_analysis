import os

import numpy as np
import pandas as pd

from csi_analysis.pipelines.scan import MaskType, FeatureExtractor, ReportGenerator

from csi_images.csi_scans import Scan
from csi_images.csi_events import EventArray


class FocusQualityExtractor(FeatureExtractor, ReportGenerator):
    """
    A pared-down "feature extractor" that extracts the quality of the
    DAPI and CD45 channels for each event found in a tile.
    """

    def __init__(self, scan: Scan, threshold: float = 0.05, save: bool = False):
        self.scan = scan
        self.threshold = threshold

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.threshold})"

    def extract_features(
        self,
        events: EventArray,
        images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
    ) -> EventArray:
        """
        Extracts the quality of each of the channels for each event in the EventArray
        through a simple gradient-based method, centered on each event.
        :param events:
        :param images:
        :param masks:
        :return:
        """
        # Scale the threshold to the image's bit depth
        if np.issubdtype(images[0].dtype, np.unsignedinteger):
            self.threshold *= np.iinfo(images[0].dtype).max

        focus_features = pd.DataFrame()
        for image, channel_name in zip(images, self.scan.get_channel_names()):
            image = image.copy()  # Avoid modifying the original image!!!
            # Threshold the image
            image[image < self.threshold] = 0
            # Calculate the gradient
            x_gradient = np.abs(np.diff(image, axis=1))
            y_gradient = np.abs(np.diff(image, axis=0)).T
            # Take the gradients at the events' locations
            x_gradient = x_gradient[events.info["x"], :]
            y_gradient = y_gradient[:, events.info["y"]]
            # Set 0 gradients to NaN to ignore them
            x_gradient[x_gradient == 0] = np.nan
            y_gradient[y_gradient == 0] = np.nan
            # Average each row of the gradients
            x_focus = np.nanmean(x_gradient[x_gradient > 0], axis=1).astype(np.float16)
            y_focus = np.nanmean(y_gradient[y_gradient > 0], axis=0).astype(np.float16)
            # Average the x and y gradients, then scale
            avg_focus = ((x_focus + y_focus) / 2 * 100).astype(np.float16)
            focus_features.loc[f"{channel_name.lower()}_quality"] = avg_focus

        events.add_features(focus_features)
        return events

    def make_report(self, events: EventArray, output_path: str) -> bool:
        """
        Save the frame info to tiles.csv with per-tile count and channel quality.
        :param events: EventArray for the whole scan with [channel]_quality in metadata
        :param output_path: Folder to save tiles.csv in
        """
        n_tiles = self.scan.roi[0].tile_rows * self.scan.roi[0].tile_cols
        # Determine the number of events per tile
        output = pd.DataFrame(
            {"count": [sum(events.info["tile"] == i) for i in range(n_tiles)]}
        )
        # Average quality in each channel for each tile
        for channel in self.scan.get_channel_names():
            channel = channel.lower()
            output[f"{channel}_quality"] = [
                np.mean(events.metadata[f"{channel}_quality"][events.info["tile"] == i])
                for i in range(n_tiles)
            ]

        output.to_csv(os.path.join(output_path, "tiles.csv"))
