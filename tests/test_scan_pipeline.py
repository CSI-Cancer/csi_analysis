import os.path
import sys

import numpy as np
import pandas as pd

import functools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from csi_images.csi_scans import Scan
from csi_images.csi_events import EventArray
from csi_analysis.pipelines.scan import *


class DummyPreprocessor(TilePreprocessor):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def preprocess(self, images: list[np.ndarray]) -> list[np.ndarray]:
        return images


class DummySegmenter(TileSegmenter):
    mask_type = MaskType.EVENT

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def segment(
        self,
        images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
    ) -> dict[MaskType, np.ndarray]:
        mask = np.zeros(images[0].shape).astype(np.uint16)
        mask[100:200, 100:200] = 1
        return {MaskType.EVENT: mask}


class DummyImageFilter(ImageFilter):
    mask_type = MaskType.EVENT

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def filter_images(
        self,
        images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
    ) -> dict[MaskType, np.ndarray]:
        return masks


class DummyFeatureExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def extract_features(
        self,
        events: EventArray,
        images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
    ) -> EventArray:
        events.add_features(pd.DataFrame({"mean_intensity": [np.mean(images[0])]}))
        return events


class DummyFeatureFilter(FeatureFilter):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def filter_features(self, events: EventArray) -> tuple[EventArray, EventArray]:
        return events, EventArray()


class DummyClassifier(EventClassifier):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def classify_events(self, events: EventArray) -> EventArray:
        events.add_metadata(
            pd.DataFrame(
                {f"model_classification{len(events)}": ["dummy"] * len(events)}
            )
        )
        return events


def test_scan_pipeline():
    scan = Scan.load_yaml("tests/data")
    log_options = {
        sys.stderr: {"level": "DEBUG", "colorize": True},
    }
    # The test should run as quickly as possible, so we use the largest possible border
    border_size = min([scan.roi[0].tile_rows / 2, scan.roi[0].tile_cols / 2])
    border_size = int(border_size - 0.5)  # Ensure that there's at least 1 valid tile
    n_tiles = (scan.roi[0].tile_rows - 2 * border_size) * (
        scan.roi[0].tile_cols - 2 * border_size
    )
    pipeline = ScanPipeline(
        scan,
        output_path="tests/data",
        preprocessors=[DummyPreprocessor()],
        segmenters=[DummySegmenter()],
        image_filters=[DummyImageFilter()],
        feature_extractors=[DummyFeatureExtractor()],
        tile_feature_filters=[DummyFeatureFilter()],
        tile_event_classifiers=[DummyClassifier()],
        scan_feature_filters=[DummyFeatureFilter()],
        scan_event_classifiers=[DummyClassifier()],
        excluded_border_size=border_size,
        save_steps=True,
        clean_steps=True,
        executor_constructor=functools.partial(
            ProcessPoolExecutor,
            max_workers=4,
            mp_context=multiprocessing.get_context("spawn"),
        ),
        log_options=log_options,
    )
    events = pipeline.run()
    assert len(events) == n_tiles
    assert os.path.exists(f"tests/data/{scan.slide_id}.hdf5")

    # Clean up
    os.remove(f"tests/data/{scan.slide_id}.hdf5")


if __name__ == "__main__":
    test_scan_pipeline()
