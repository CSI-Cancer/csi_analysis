import logging

import numpy as np
import pandas as pd

from csi_analysis.utils import csi_logging
from csi_images import csi_scans, csi_tiles, csi_events
from csi_analysis.pipelines import scan_pipeline


class DummyPreprocessor(scan_pipeline.TilePreprocessor):
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        verbose: bool = False,
        save: bool = False,
    ):
        """
        Must have a logging.Logger as self.log.
        :param scan: scan metadata, which may be used for inferring parameters.
        :param version: a version string, recommended to be an ISO date.
        :param save: whether to save the immediate results of this module.
        """
        self.scan = scan
        self.version = version
        self.save = save
        self.verbose = verbose
        self.log = csi_logging.get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version})"

    def preprocess(self, frame_images: list[np.ndarray]) -> list[np.ndarray]:
        return frame_images


class DummySegmenter(scan_pipeline.TileSegmenter):
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        verbose: bool = False,
        save: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save = save
        self.verbose = verbose
        self.log = csi_logging.get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )
        # List of output mask types that this segmenter can output; must exist
        self.mask_types = [mask_type for mask_type in scan_pipeline.MaskType]

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version})"

    def segment(
        self, frame_images: list[np.ndarray]
    ) -> dict[scan_pipeline.MaskType, np.ndarray]:
        mask = np.zeros(frame_images[0].shape).astype(np.uint16)
        mask[100:200, 100:200] = 1
        return {scan_pipeline.MaskType.EVENT: mask}


class DummyImageFilter(scan_pipeline.ImageFilter):
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        verbose: bool = False,
        save: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save = save
        self.verbose = verbose
        self.log = csi_logging.get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version})"

    def filter_images(
        self,
        frame_images: list[np.ndarray],
        masks: dict[scan_pipeline.MaskType, np.ndarray],
    ) -> dict[scan_pipeline.MaskType, np.ndarray]:
        return masks


class DummyFeatureExtractor(scan_pipeline.FeatureExtractor):
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        verbose: bool = False,
        save: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save = save
        self.verbose = verbose
        self.log = csi_logging.get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version})"

    def extract_features(
        self,
        frame_images: list[np.ndarray],
        masks: dict[scan_pipeline.MaskType, np.ndarray],
        events: csi_events.EventArray,
    ) -> pd.DataFrame:
        features = pd.DataFrame({"mean_intensity": [np.mean(frame_images[0])]})
        return features


class DummyFeatureFilter(scan_pipeline.FeatureFilter):
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        verbose: bool = False,
        save: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save = save
        self.verbose = verbose
        self.log = csi_logging.get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version})"

    def filter_features(
        self, events: csi_events.EventArray
    ) -> tuple[csi_events.EventArray, csi_events.EventArray]:
        return events, csi_events.EventArray()


class DummyClassifier(scan_pipeline.EventClassifier):
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        verbose: bool = False,
        save: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save = save
        self.verbose = verbose
        self.log = csi_logging.get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version})"

    def classify_events(self, events: csi_events.EventArray) -> csi_events.EventArray:
        events.add_metadata(
            pd.DataFrame(
                {f"model_classification{len(events)}": ["dummy"] * len(events)}
            )
        )
        return events


def test_scan_pipeline():
    scan = csi_scans.Scan.load_yaml("tests/data")
    pipeline = scan_pipeline.TilingScanPipeline(
        scan,
        output_path="tests/data",
        preprocessors=[DummyPreprocessor(scan, "2024-10-30")],
        segmenters=[DummySegmenter(scan, "2024-10-30")],
        image_filters=[DummyImageFilter(scan, "2024-10-30")],
        feature_extractors=[DummyFeatureExtractor(scan, "2024-10-30")],
        tile_feature_filters=[DummyFeatureFilter(scan, "2024-10-30")],
        tile_event_classifiers=[DummyClassifier(scan, "2024-10-30")],
        scan_feature_filters=[DummyFeatureFilter(scan, "2024-10-30")],
        scan_event_classifiers=[DummyClassifier(scan, "2024-10-30")],
        verbose=True,
        max_workers=1,
    )
    events = pipeline.run()
    assert len(events) == scan.roi[0].tile_rows * scan.roi[0].tile_cols
