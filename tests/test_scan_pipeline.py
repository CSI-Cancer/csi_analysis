import numpy as np
import pandas as pd
from csi_images import csi_scans, csi_tiles, csi_events

from csi_analysis import csi_scan_pipeline
from csi_analysis.csi_scan_pipeline import TileSegmenter


class DummyPreprocessor(csi_scan_pipeline.TilePreprocessor):
    def __init__(self, scan: csi_scans.Scan, version: str, save_output: bool = False):
        self.scan = scan
        self.version = version
        self.save_output = save_output

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    def preprocess(self, frame_images: list[np.ndarray]) -> list[np.ndarray]:
        return frame_images


class DummySegmenter(csi_scan_pipeline.TileSegmenter):
    def __init__(self, scan: csi_scans.Scan, version: str, save_output: bool = False):
        self.scan = scan
        self.version = version
        self.save_output = save_output

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    def segment(
        self, frame_images: list[np.ndarray]
    ) -> dict[csi_scan_pipeline.MaskType, np.ndarray]:
        return {csi_scan_pipeline.MaskType.EVENT: np.zeros(frame_images[0].shape)}


class DummyImageFilter(csi_scan_pipeline.ImageFilter):
    def __init__(self, scan: csi_scans.Scan, version: str, save_output: bool = False):
        self.scan = scan
        self.version = version
        self.save_output = save_output

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    def filter_images(
        self,
        frame_images: list[np.ndarray],
        masks: dict[csi_scan_pipeline.MaskType, np.ndarray],
    ) -> dict[csi_scan_pipeline.MaskType, np.ndarray]:
        return masks


class DummyFeatureExtractor(csi_scan_pipeline.FeatureExtractor):
    def __init__(self, scan: csi_scans.Scan, version: str, save_output: bool = False):
        self.scan = scan
        self.version = version
        self.save_output = save_output

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    def extract_features(
        self,
        frame_images: list[np.ndarray],
        masks: dict[csi_scan_pipeline.MaskType, np.ndarray],
    ) -> csi_events.EventArray:
        event = csi_events.Event(
            self.scan,
            csi_tiles.Tile(self.scan, 0),
            10,
            20,
            30,
            features=pd.Series({"mean_intensity": [np.mean(frame_images[0])]}),
        )
        return csi_events.EventArray.from_events([event])


class DummyFeatureFilter(csi_scan_pipeline.FeatureFilter):
    def __init__(self, scan: csi_scans.Scan, version: str, save_output: bool = False):
        self.scan = scan
        self.version = version
        self.save_output = save_output

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    def filter_features(
        self, events: csi_events.EventArray
    ) -> tuple[csi_events.EventArray, csi_events.EventArray]:
        return events, csi_events.EventArray()


class DummyClassifier(csi_scan_pipeline.EventClassifier):
    def __init__(self, scan: csi_scans.Scan, version: str, save_output: bool = False):
        self.scan = scan
        self.version = version
        self.save_output = save_output

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    def classify_events(self, events: csi_events.EventArray) -> csi_events.EventArray:
        events.add_metadata(
            pd.DataFrame(
                {f"model_classification{len(events)}": ["dummy"] * len(events)}
            )
        )
        return events


def test_scan_pipeline():
    scan = csi_scans.Scan.load_yaml("tests/data")
    pipeline = csi_scan_pipeline.ScanPipeline(
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
    )
    events = pipeline.run()
    assert len(events) == scan.roi[0].tile_rows * scan.roi[0].tile_cols
