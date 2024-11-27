import os.path

import numpy as np

from csi_images.csi_events import EventArray
from csi_images.csi_scans import Scan

from ..pipelines.scan_pipeline import ReportGenerator


class OCULARReportGenerator(ReportGenerator):

    def __init__(self, scan: Scan, save: bool = False):
        self.scan = scan
        self.save = save

    def make_report(
        self,
        output_path: str,
        events: EventArray,
        images: list[list[np.ndarray]] = None,
    ) -> bool:
        # Create dummy files
        for file in ["out.rds", "cc-final.csv", "others-final.csv"]:
            open(file, "a").close()
        # Confirm all files were created
        success = True
        for file in ["out.rds", "cc-final.csv", "others-final.csv"]:
            success = success and os.path.exists(file)
        return success

    def __repr__(self):
        return f"{self.__class__.__name__})"
