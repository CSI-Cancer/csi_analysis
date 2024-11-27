import os
import warnings

import numpy as np
import pandas as pd
from csi_images.csi_scans import Scan
from csi_images.csi_tiles import Tile
from csi_images.csi_frames import Frame
from csi_images.csi_events import EventArray


class OCULARFrameInfoGenerator:
    """
    A pared-down "feature extractor" that extracts the quality of the
    DAPI and CD45 channels for each event found in a tile.
    """

    def __init__(self, scan: Scan, threshold=0.05, save: bool = False):
        self.scan = scan
        self.dapi_idx, self.cy5_idx = scan.get_channel_indices(["DAPI", "AF647"])
        self.threshold = threshold
        self.save = save

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.threshold})"

    def extract_event_quality(self, x, y, dapi, cy5) -> tuple[float, float]:
        """
        Finds the quality of the DAPI and "CY5" channels for a single event.
        :param x: x-coordinate of the event
        :param y: y-coordinate of the event
        :param dapi: DAPI channel image
        :param cy5: CD45 channel image
        :return: a tuple of the DAPI and CD45 quality
        """
        # DAPI gradient at same x
        dapi_x = dapi[:, x] / 65535
        dapi_x[dapi_x < self.threshold] = 0
        dapi_x = np.abs(np.diff(dapi_x))
        dapi_x = dapi_x[dapi_x > 0]
        if len(dapi_x) == 0:
            dapi_x = 0
        else:
            dapi_x = np.mean(dapi_x) * 100
        # DAPI gradient at same y
        dapi_y = dapi[y, :] / 65535
        dapi_y[dapi_y < self.threshold] = 0
        dapi_y = np.abs(np.diff(dapi_y))
        dapi_y = dapi_y[dapi_y > 0]
        if len(dapi_y) == 0:
            dapi_y = 0
        else:
            dapi_y = np.mean(dapi_y) * 100
        # Average DAPI quality
        dapi = np.mean([dapi_x, dapi_y])
        # CD45 gradient at same x
        cy5_x = cy5[:, x] / 65535
        cy5_x[cy5_x < self.threshold] = 0
        cy5_x = np.abs(np.diff(cy5_x))
        cy5_x = cy5_x[cy5_x > 0]
        if len(cy5_x) == 0:
            cy5_x = 0
        else:
            cy5_x = np.mean(cy5_x) * 100
        # CD45 gradient at same y
        cy5_y = cy5[y, :] / 65535
        cy5_y[cy5_y < self.threshold] = 0
        cy5_y = np.abs(np.diff(cy5_y))
        cy5_y = cy5_y[cy5_y > 0]
        if len(cy5_y) == 0:
            cy5_y = 0
        else:
            cy5_y = np.mean(cy5_y) * 100
        # Average CD45 quality
        cy5 = np.mean([cy5_x, cy5_y])
        return np.float16(dapi), np.float16(cy5)

    def extract_tile_quality(
        self, events: EventArray, images: list[np.ndarray] = None
    ) -> EventArray:
        """
        Finds the quality of the DAPI and "CY5" channels for each event in a tile
        :param events: EventArray for one tile
        :param images: list of numpy arrays representing each channel; will load if None
        :return: an EventArray with "dapi_quality" and "cy5_quality" in metadata
        """
        # Copy to avoid modifying the original EventArray
        events = events.copy()
        events.metadata["dapi_quality"] = np.zeros(len(events), dtype=np.float16)
        events.metadata["cy5_quality"] = np.zeros(len(events), dtype=np.float16)
        # Populate images if needed
        if images is not None:
            dapi = images[self.dapi_idx]
            cy5 = images[self.cy5_idx]
        else:
            tile = Tile(self.scan, events.info["tile"][0])
            dapi = Frame.get_frames(tile, (self.dapi_idx,))[0].get_image()
            cy5 = Frame.get_frames(tile, (self.cy5_idx,))[0].get_image()
        # Loop through events, populating metadata
        for i in range(len(events)):
            # Determine the (x, y) coordinates of the event
            x = events.info["x"][i]
            y = events.info["y"][i]
            dapi_quality, cy5_quality = self.extract_event_quality(x, y, dapi, cy5)
            events.metadata.loc[i, "dapi_quality"] = dapi_quality
            events.metadata.loc[i, "cy5_quality"] = cy5_quality
        return events

    def extract_scan_quality(
        self, events: EventArray, images: list[list[np.ndarray]] = None
    ) -> EventArray:
        """
        Contrary to normal feature extractors, this extracts image quality as metadata
        :param events: EventArray for full scan
        :param images: list of numpy arrays representing each channel; will load if None
        :return: an EventArray with "dapi_quality" and "cy5_quality" in metadata
        """
        # Copy to avoid modifying the original EventArray
        events = events.copy()
        events.metadata["dapi_quality"] = np.zeros(len(events), dtype=np.float16)
        events.metadata["cy5_quality"] = np.zeros(len(events), dtype=np.float16)
        # Loop through each tile
        for i in range(len(self.scan.roi[0].tile_rows * self.scan.roi[0].tile_cols)):
            rows = events.info["tile"] == i
            # Skip if no relevant events
            if sum(rows) == 0:
                continue
            # Load in images
            if images is not None:
                dapi = images[i][self.dapi_idx]
                cy5 = images[i][self.cy5_idx]
            else:
                tile = Tile(self.scan, i)
                tile_images = [None] * len(self.scan.channels)
                tile_images[self.dapi_idx] = Frame.get_frames(tile, (self.dapi_idx,))[0]
                tile_images[self.dapi_idx] = tile_images[self.dapi_idx].get_image()
                tile_images[self.cy5_idx] = Frame.get_frames(tile, (self.cy5_idx,))[0]
                tile_images[self.cy5_idx] = tile_images[self.cy5_idx].get_image()
            # Get quality
            tile_events = events.rows(rows)
            tile_events = self.extract_tile_quality(tile_events, tile_images)
            # Move it into the main EventArray
            events.metadata.loc[rows, "dapi_quality"] = tile_events.metadata[
                "dapi_quality"
            ]
            events.metadata.loc[rows, "cy5_quality"] = tile_events.metadata[
                "cy5_quality"
            ]
        return events

    def save_frameinfo_csv(self, output_path: str, events: EventArray) -> bool:
        """
        Save the frame info to frameinfo.csv, in the OCULAR format.
        :param output_path: Folder to save frameinfo.csv in
        :param events: EventArray for the whole scan with
        "dapi_quality" and "cy5_quality" in metadata
        """
        # Reorganize the required metadata in frameinfo.csv
        # Generate a dataframe with one row for each tile
        df = pd.DataFrame(
            {
                "frame_id": list(
                    range(self.scan.roi[0].tile_rows * self.scan.roi[0].tile_cols)
                )
            }
        )
        # Gather the number of events in each tile
        df["cell_count"] = [sum(events.info["tile"] == i) for i in df["frame_id"]]
        # Gather the average quality of DAPI and CD45 in each tile, ignoring
        # empty slice warnings (some frames may have no events)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df["dapi_quality"] = [
                np.mean(events.metadata["dapi_quality"][events.info["tile"] == i])
                for i in df["frame_id"]
            ]
            df["cy5_quality"] = [
                np.mean(events.metadata["cy5_quality"][events.info["tile"] == i])
                for i in df["frame_id"]
            ]
            df = df.fillna(0)
        # Find the average
        df["avg_quality"] = (df["dapi_quality"] + df["cy5_quality"]) / 2
        # Round to 3 decimal places
        df = df.round(3)
        # Classically 1-indexed because of R
        df["frame_id"] += 1
        df.to_csv(os.path.join(output_path, "frameinfo.csv"))
