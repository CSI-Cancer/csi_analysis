import os
import time
import logging
import logging.handlers

from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

multiprocessing.set_start_method("spawn", force=True)

from csi_analysis import csi_logging
from csi_images import csi_scans, csi_tiles, csi_frames, csi_events


class MaskType(Enum):
    EVENT = "event"
    DAPI_ONLY = "dapi_only"
    CELLS_ONLY = "cells_only"
    OTHERS_ONLY = "others_only"
    STAIN_ARTIFACT = "stain_artifact"
    SLIDE_ARTIFACT = "slide_artifact"
    SCAN_ARTIFACT = "scan_artifact"
    OTHER = "other"


class TilePreprocessor(ABC):
    """
    Abstract class for a tile preprocessor.
    """

    @abstractmethod
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        save_output: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save_output = save_output
        self.log = csi_logging.get_logger()

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @abstractmethod
    def preprocess(self, frame_images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Preprocess the frames of a tile. Should return the frames in the same order.
        No coordinate system changes should occur here, as they are handled elsewhere.
        :param frame_images: a list of np.ndarrays, each representing a frame.
        :return: a list of np.ndarrays, each representing a frame.
        """
        pass


class TileSegmenter(ABC):
    """
    Abstract class for a tile segmenter.
    """

    @abstractmethod
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        save_output: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save_output = save_output
        self.log = csi_logging.get_logger()

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @abstractmethod
    def segment(self, frame_images: list[np.ndarray]) -> dict[MaskType, np.ndarray]:
        """
        Segments the frames of a tile to enumerated mask(s). Mask(s) should be returned
        in a dict with labeled types.
        :param frame_images: a list of np.ndarrays, each representing a frame.
        :return: a dict of np.ndarrays, each representing a mask.
        """
        pass


class ImageFilter(ABC):
    """
    Abstract class for an image-based event filter.
    """

    @abstractmethod
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        save_output: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save_output = save_output
        self.log = csi_logging.get_logger()

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @abstractmethod
    def filter_images(
        self,
        frame_images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
    ) -> dict[MaskType, np.ndarray]:
        """
        Removes objects from important masks based on other masks or new analysis.
        :param frame_images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :return: a dict of np.ndarrays, each representing a mask; now filtered.
        """
        pass


class FeatureExtractor(ABC):
    """
    Abstract class for a feature extractor.
    """

    @abstractmethod
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        save_output: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save_output = save_output
        self.log = csi_logging.get_logger()

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @abstractmethod
    def extract_features(
        self,
        frame_images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
    ) -> csi_events.EventArray:
        """
        Segments the frames of a tile to enumerated mask(s). Mask(s) should be returned
        in a dict with labeled types.
        :param frame_images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :return: a csi_events.EventArray with populated features.
        """
        pass


class FeatureFilter(ABC):
    """
    Abstract class for a feature-based event filter.
    """

    @abstractmethod
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        save_output: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save_output = save_output
        self.log = csi_logging.get_logger()

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @abstractmethod
    def filter_features(
        self,
        events: csi_events.EventArray,
    ) -> tuple[csi_events.EventArray, csi_events.EventArray]:
        """
        Removes events from an event array based on feature values.
        :param events: a csi_events.EventArray with populated features.
        :return: two csi_events.EventArray objects: tuple[remaining, filtered]
        """
        pass


class EventClassifier(ABC):
    """
    Abstract class for an event classifier.
    """

    @abstractmethod
    def __init__(
        self,
        scan: csi_scans.Scan,
        version: str,
        save_output: bool = False,
    ):
        self.scan = scan
        self.version = version
        self.save_output = save_output
        self.log = csi_logging.get_logger()

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @abstractmethod
    def classify_events(
        self,
        events: csi_events.EventArray,
    ) -> csi_events.EventArray:
        """
        Classifies events based on features, then populates the metadata.
        :param events: a csi_events.EventArray with populated features.
        :return: a csi_events.EventArray with populated metadata.
        """
        pass


class ScanPipeline:
    """
    An image-based analysis pipeline for scans.
    Here, we assume that tiles of the scan cannot be stitched together, nor is
    it desired to do so. Instead, we perform image tasks on the tiles separately.

    However, we do assume that events from different tiles can be stitched together and
    analyzed as a whole, so we allow for event filtering and classification at both
    the tile and scan levels.
    """

    def __init__(
        self,
        scan: csi_scans.Scan,
        output_path: str,
        preprocessors: list[TilePreprocessor] = None,
        segmenters: list[TileSegmenter] = None,
        image_filters: list[ImageFilter] = None,
        feature_extractors: list[FeatureExtractor] = None,
        tile_feature_filters: list[FeatureFilter] = None,
        tile_event_classifiers: list[EventClassifier] = None,
        scan_feature_filters: list[FeatureFilter] = None,
        scan_event_classifiers: list[EventClassifier] = None,
        save_steps: bool = False,
        max_workers: int = 61,
        verbose: bool = False,
    ):
        self.scan = scan
        self.output_path = output_path
        self.save_steps = save_steps
        self.max_workers = max_workers
        self.verbose = verbose
        if preprocessors is None:
            preprocessors = []
        elif isinstance(preprocessors, TilePreprocessor):
            preprocessors = [preprocessors]
        self.preprocessors = preprocessors
        if segmenters is None:
            segmenters = []
        elif isinstance(segmenters, TileSegmenter):
            segmenters = [segmenters]
        self.segmenters = segmenters
        if image_filters is None:
            image_filters = []
        elif isinstance(image_filters, ImageFilter):
            image_filters = [image_filters]
        self.image_filters = image_filters
        if feature_extractors is None:
            feature_extractors = []
        elif isinstance(feature_extractors, FeatureExtractor):
            feature_extractors = [feature_extractors]
        self.feature_extractors = feature_extractors
        if tile_feature_filters is None:
            tile_feature_filters = []
        elif isinstance(tile_feature_filters, FeatureFilter):
            tile_feature_filters = [tile_feature_filters]
        self.tile_feature_filters = tile_feature_filters
        if tile_event_classifiers is None:
            tile_event_classifiers = []
        elif isinstance(tile_event_classifiers, EventClassifier):
            tile_event_classifiers = [tile_event_classifiers]
        self.tile_event_classifiers = tile_event_classifiers
        if scan_feature_filters is None:
            scan_feature_filters = []
        elif isinstance(scan_feature_filters, FeatureFilter):
            scan_feature_filters = [scan_feature_filters]
        self.scan_feature_filters = scan_feature_filters
        if scan_event_classifiers is None:
            scan_event_classifiers = []
        elif isinstance(scan_event_classifiers, EventClassifier):
            scan_event_classifiers = [scan_event_classifiers]
        self.scan_event_classifiers = scan_event_classifiers

    def run(self) -> csi_events.EventArray:
        """
        Runs the pipeline on the scan.
        """
        # Set up logger
        log = csi_logging.get_logger(
            level=logging.DEBUG if self.verbose else logging.INFO,
            log_file=os.path.join(self.output_path, "pipeline.log"),
        )

        log.info("Beginning to run the pipeline on the scan")
        # Get all tiles
        tiles = csi_tiles.Tile.get_tiles(self.scan)
        # First, do tile-specific steps
        max_workers = min(multiprocessing.cpu_count() - 1, 61)
        if self.max_workers <= 1:
            events = list(
                map(
                    execute_pipeline_on_tile,
                    itertools.repeat(self.preprocessors),
                    itertools.repeat(self.segmenters),
                    itertools.repeat(self.image_filters),
                    itertools.repeat(self.feature_extractors),
                    itertools.repeat(self.tile_feature_filters),
                    itertools.repeat(self.tile_event_classifiers),
                    tiles,
                    itertools.repeat(self.verbose),
                    itertools.repeat(self.save_steps),
                )
            )
        else:
            log_queue = csi_logging.start_multiprocess_logging(
                log, using_ProcessPoolExecutor=True
            )
            with ProcessPoolExecutor(max_workers) as executor:
                events = list(
                    executor.map(
                        execute_pipeline_on_tile,
                        itertools.repeat(self.preprocessors),
                        itertools.repeat(self.segmenters),
                        itertools.repeat(self.image_filters),
                        itertools.repeat(self.feature_extractors),
                        itertools.repeat(self.tile_feature_filters),
                        itertools.repeat(self.tile_event_classifiers),
                        tiles,
                        itertools.repeat(self.save_steps),
                        itertools.repeat(self.verbose),
                        itertools.repeat(log_queue),
                    )
                )
            csi_logging.stop_multiprocess_logging()

        # Then, do overall steps
        # Combine EventArrays from all tiles
        events = csi_events.EventArray.from_list(events)

        # Filter events by features at the scan level
        for feature_filter in self.scan_feature_filters:
            start_time = time.time()
            log.debug(f"Running feature filter {feature_filter.__repr__()} on the scan")
            events, filtered = feature_filter.filter_features(events)
            log.debug(
                f"Feature filter {feature_filter.__repr__()} on the scan "
                f"took {time.time() - start_time} seconds"
            )
            if self.save_steps and feature_filter.save_output:
                log.debug(
                    f"Saving output for feature filter {feature_filter.__repr__()} "
                    f"on the scan"
                )
                log.warning("Saving output for feature filter not yet implemented")

        # Classify events at the tile level
        if len(self.scan_event_classifiers) > 1:
            log.warning(
                "Multiple event classifiers may overwrite each other's metadata"
            )
        for event_classifier in self.scan_event_classifiers:
            start_time = time.time()
            log.debug(
                f"Running event classifier {event_classifier.__repr__()} on the scan"
            )
            events = event_classifier.classify_events(events)
            log.debug(
                f"Event classifier {event_classifier.__repr__()} on the scan "
                f"took {time.time() - start_time} seconds"
            )
            if self.save_steps and event_classifier.save_output:
                log.debug(
                    f"Saving output for event classifier {event_classifier.__repr__()} "
                    f"on the scan"
                )
                log.warning("Saving output for event classifier not yet implemented")

        log.info("Finished running the pipeline on the scan")

        return events


def execute_pipeline_on_tile(
    preprocessors: list[TilePreprocessor],
    segmenters: list[TileSegmenter],
    image_filters: list[ImageFilter],
    feature_extractors: list[FeatureExtractor],
    feature_filters: list[FeatureFilter],
    event_classifiers: list[EventClassifier],
    tile: csi_tiles.Tile,
    save_steps: bool = False,
    verbose: bool = False,
    log_queue: multiprocessing.Queue = None,
):
    """
    Runs tile-specific pipeline steps on a tile.
    :param preprocessors:
    :param segmenters:
    :param image_filters:
    :param feature_extractors:
    :param feature_filters:
    :param event_classifiers:
    :param tile: the tile to run the pipeline on.
    :param save_steps:
    :param verbose:
    :param log_queue: a multiprocessing.Queue to log to.
    :return: a csi_events.EventArray with populated features and potentially
             populated metadata.
    """
    log = csi_logging.get_logger(
        name="csi_tile_pipeline",
        level=logging.DEBUG if verbose else logging.INFO,
        queue=log_queue,
    )
    # Load the tile frames
    frame_images = [frame.get_image()[0] for frame in csi_frames.Frame.get_frames(tile)]
    log.debug(f"Loaded {len(frame_images)} frame images for tile {tile.n}")

    # Preprocess the frames
    for preprocessor in preprocessors:
        start_time = time.time()
        log.debug(f"Running preprocessor {preprocessor.__repr__()} on tile {tile.n}")
        if log_queue is not None and hasattr(preprocessor, "log"):
            preprocessor.log.addHandler(logging.handlers.QueueHandler(log_queue))
        frame_images = preprocessor.preprocess(frame_images)
        log.debug(
            f"Preprocessor {preprocessor.__repr__()} on tile {tile.n} "
            f"took {time.time() - start_time} seconds"
        )
        if save_steps and preprocessor.save_output:
            log.debug(
                f"Saving output for preprocessor {preprocessor.__repr__()} "
                f"on tile {tile.n}"
            )
            log.warning("Saving output for segmenter not yet implemented")

    # Segment the frames
    masks = {}
    for segmenter in segmenters:
        start_time = time.time()
        log.debug(f"Running segmenter {segmenter.__repr__()} on tile {tile.n}")
        if log_queue is not None and hasattr(segmenter, "log"):
            segmenter.log.addHandler(logging.handlers.QueueHandler(log_queue))
        new_masks = segmenter.segment(frame_images)
        for key in new_masks:
            if key in masks:
                log.warning(f"{key} mask has already been populated; ignoring")
            else:
                masks[key] = new_masks[key]
        log.debug(
            f"Segmenter {segmenter.__repr__()} on tile {tile.n} "
            f"took {time.time() - start_time} seconds"
        )
        if save_steps and segmenter.save_output:
            log.debug(
                f"Saving output for segmenter {segmenter.__repr__()} "
                f"on tile {tile.n}"
            )
            log.warning("Saving output for segmenter not yet implemented")

    # Filter the images
    for image_filter in image_filters:
        start_time = time.time()
        log.debug(f"Running image filter {image_filter.__repr__()} on tile {tile.n}")
        if log_queue is not None and hasattr(image_filter, "log"):
            image_filter.log.addHandler(logging.handlers.QueueHandler(log_queue))
        masks = image_filter.filter_images(frame_images, masks)
        log.debug(
            f"Image filter {image_filter.__repr__()} on tile {tile.n} "
            f"took {time.time() - start_time} seconds"
        )
        if save_steps and image_filter.save_output:
            log.debug(
                f"Saving output for image filter {image_filter.__repr__()} "
                f"on tile {tile.n}"
            )
            log.warning("Saving output for image filter not yet implemented")

    # Extract features
    eventarray_list = []
    for feature_extractor in feature_extractors:
        start_time = time.time()
        log.debug(
            f"Running feature extractor {feature_extractor.__repr__()} on tile {tile.n}"
        )
        if log_queue is not None and hasattr(feature_extractor, "log"):
            feature_extractor.log.addHandler(logging.handlers.QueueHandler(log_queue))
        eventarray_list.append(feature_extractor.extract_features(frame_images, masks))
        log.debug(
            f"Feature extractor {feature_extractor.__repr__()} on tile {tile.n} "
            f"took {time.time() - start_time} seconds"
        )
        if save_steps and feature_extractor.save_output:
            log.debug(
                f"Saving output for feature extractor {feature_extractor.__repr__()} "
                f"on tile {tile.n}"
            )
            log.warning("Saving output for feature extractor not yet implemented")
    if len(eventarray_list) > 1:
        log.debug(f"Concatenating event features for tile {tile.n}")
        events = eventarray_list[0].features.join(
            [features for features in eventarray_list[1:]]
        )
    else:
        events = eventarray_list[0]

    # Filter events by features at the tile level
    for feature_filter in feature_filters:
        start_time = time.time()
        log.debug(
            f"Running feature filter {feature_filter.__repr__()} on tile {tile.n}"
        )
        if log_queue is not None and hasattr(feature_filter, "log"):
            feature_filter.log.addHandler(logging.handlers.QueueHandler(log_queue))
        events, filtered = feature_filter.filter_features(events)
        log.debug(
            f"Feature filter {feature_filter.__repr__()} on tile {tile.n} "
            f"took {time.time() - start_time} seconds"
        )
        if save_steps and feature_filter.save_output:
            log.debug(
                f"Saving output for feature filter {feature_filter.__repr__()} "
                f"on tile {tile.n}"
            )
            log.warning("Saving output for feature filter not yet implemented")

    # Classify events at the tile level
    if len(event_classifiers) > 1:
        log.warning("Multiple event classifiers may overwrite each other's metadata")
    for event_classifier in event_classifiers:
        start_time = time.time()
        log.debug(
            f"Running event classifier {event_classifier.__repr__()} on tile {tile.n}"
        )
        if log_queue is not None and hasattr(event_classifier, "log"):
            event_classifier.log.addHandler(logging.handlers.QueueHandler(log_queue))
        events = event_classifier.classify_events(events)
        log.debug(
            f"Event classifier {event_classifier.__repr__()} on tile {tile.n} "
            f"took {time.time() - start_time} seconds"
        )
        if save_steps and event_classifier.save_output:
            log.debug(
                f"Saving output for event classifier {event_classifier.__repr__()} "
                f"on tile {tile.n}"
            )
            log.warning("Saving output for event classifier not yet implemented")

    return events
