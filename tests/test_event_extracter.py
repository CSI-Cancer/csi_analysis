import logging

import numpy as np
import pandas as pd

from csi_analysis.utils import csi_logging
from csi_images import csi_scans, csi_tiles, csi_events
from csi_analysis.modules import event_extracter


def test_scan_pipeline():
    scan = csi_scans.Scan.load_yaml("tests/data")
    tile = csi_tiles.Tile(scan, 100)
    mask = np.load("tests/data/mask.npy")
    events = event_extracter.mask_to_events(scan, tile, mask)
    assert isinstance(events, csi_events.EventArray)
