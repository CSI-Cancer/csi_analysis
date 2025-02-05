import os
import warnings

import numpy as np
import pandas as pd

from csi_analysis.pipelines.scan import MaskType, FeatureExtractor, ReportGenerator

from csi_images.csi_scans import Scan
from csi_images.csi_events import EventArray
from csi_images import csi_images


class FocusQualityExtractor(FeatureExtractor, ReportGenerator):
    """
    A pared-down "feature extractor" that extracts the quality of the
    DAPI and CD45 channels for each event found in a tile.
    """

    def __init__(self, scan: Scan, threshold: float = 0.05):
        self.scan = scan
        self.threshold = threshold

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.threshold}"

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
        focus_features = []
        for image, channel_name in zip(images, self.scan.get_channel_names()):
            image = csi_images.scale_bit_depth(image, np.float16)
            # Threshold the image
            image[image < self.threshold] = 0
            # Calculate the gradient for each x and y position
            y_gradient = np.abs(np.diff(image, axis=1))
            x_gradient = np.abs(np.diff(image, axis=0))
            # Take the gradients at the events' positions
            y_gradient = y_gradient[events.info["y"], :]
            x_gradient = x_gradient[:, events.info["x"]]
            # Set 0 gradients to NaN to ignore them
            y_gradient[y_gradient == 0] = np.nan
            x_gradient[x_gradient == 0] = np.nan
            # Average each row of the gradients, ignoring nanmean warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y_focus = np.nanmean(y_gradient, axis=1).astype(np.float16)
                y_focus[np.isnan(y_focus)] = 0
                x_focus = np.nanmean(x_gradient, axis=0).astype(np.float16)
                x_focus[np.isnan(x_focus)] = 0
            # Average the x and y gradients, then scale
            avg_focus = ((x_focus + y_focus) / 2 * 100).astype(np.float16)
            focus_features.append(
                pd.DataFrame({f"{channel_name.lower()}_quality": avg_focus})
            )
        focus_features = pd.concat(focus_features, axis=1)
        events.add_metadata(focus_features)
        return events

    def make_report(self, events: EventArray, output_path: str) -> bool:
        """
        Save the frame info to tiles.csv with per-tile count and channel quality.
        :param events: EventArray for the whole scan with [channel]_quality in metadata
        :param output_path: Folder to save tiles.csv in
        """
        n_tiles = self.scan.roi[0].tile_rows * self.scan.roi[0].tile_cols
        # Determine the number of events per tile
        tile_counts = np.zeros(n_tiles, dtype=np.uint16)
        tiles_n, counts = np.unique(events.info["tile"], return_counts=True)
        tile_counts[tiles_n] = counts
        output = pd.DataFrame({"count": tile_counts})
        # Average quality in each channel for each tile
        for channel in self.scan.get_channel_names():
            tile_averages = np.zeros(n_tiles, dtype=np.float16)
            for i in range(n_tiles):
                if tile_counts[i] > 0:
                    tile_averages[i] = np.mean(
                        events.metadata[f"{channel.lower()}_quality"][
                            events.info["tile"] == i
                        ]
                    )
            output[f"{channel.lower()}_quality"] = tile_averages

        # Drop the quality columns from the EventArray
        events.metadata = events.metadata.drop(
            columns=[f"{c.lower()}_quality" for c in self.scan.get_channel_names()]
        )

        # Save the output to a CSV file
        output.to_csv(os.path.join(output_path, "tiles.csv"))
