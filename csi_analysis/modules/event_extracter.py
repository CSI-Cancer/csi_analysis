"""
Converts a mask to a list of events.
"""

import logging

import numpy as np
import pandas as pd

from skimage.measure import regionprops_table

from csi_images.csi_scans import Scan
from csi_images.csi_tiles import Tile
from csi_images.csi_events import EventArray


def mask_to_events(
    scan: Scan,
    tile: Tile,
    mask: np.ndarray,
    log: logging.Logger = None,
) -> EventArray:
    """
    Extracts events from a mask.
    :param scan: scan metadata
    :param tile: tile metadata
    :param mask: mask to extract events from
    :param log: optional logger
    :return: EventArray containing the set of events, corresponding to the
    labeled regions in the mask IF AND ONLY IF the mask labels are sequential
    (e.g. missing labels will make this not true).
    """
    if np.max(mask) == 0:
        # Nothing here, return an empty EventArray
        return EventArray()

    # Use skimage.measure.regionprops_table to compute properties
    info = pd.DataFrame(
        regionprops_table(
            mask,
            properties=[
                "label",
                "centroid",
                "axis_major_length",
            ],
        )
    )

    # Rename columns to match desired output
    info.rename(
        columns={
            "centroid-0": "y",
            "centroid-1": "x",
            "axis_major_length": "size",
        },
        inplace=True,
    )
    # Check the label column to see if it matches the number of events
    if max(info["label"]) != len(info["label"]) and log is not None:
        log.warn("The number of events in the mask does not match the label numbering")

    # Reorder and drop label
    info = info[["x", "y", "size"]]
    # Pad on the key metadata
    other_info = pd.DataFrame(
        {
            "slide_id": [scan.slide_id] * len(info),
            "tile": [tile.n] * len(info),
            "roi": [tile.n_roi] * len(info),
        },
    )
    info = pd.concat([other_info, info], axis=1)
    return EventArray(info, None, None)
