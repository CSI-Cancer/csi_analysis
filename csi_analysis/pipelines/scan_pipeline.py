import os
import time
import logging
import logging.handlers

import typing
from enum import Enum
from abc import ABC, abstractmethod

import cv2
import numpy as np

import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

multiprocessing.set_start_method("spawn", force=True)

from csi_images import csi_scans, csi_tiles, csi_frames, csi_events
from csi_analysis.modules import event_extracter
from csi_analysis.utils import csi_logging


class MaskType(Enum):
    EVENT = "event"
    DAPI_ONLY = "dapi_only"
    CELLS_ONLY = "cells_only"
    OTHERS_ONLY = "others_only"
    STAIN_ARTIFACT = "stain_artifact"
    SLIDE_ARTIFACT = "slide_artifact"
    SCAN_ARTIFACT = "scan_artifact"
    OTHER = "other"
    REMOVED = "removed"


class TilePreprocessor(ABC):
    """
    Abstract class for a tile preprocessor.
    """

    @abstractmethod
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

    @abstractmethod
    def preprocess(self, frame_images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Preprocess the frames of a tile, preferably in-place.
        Should return the frames in the same order.
        No coordinate system changes should occur here, as they are handled elsewhere.
        :param frame_images: a list of np.ndarrays, each representing a frame.
        :return: a list of np.ndarrays, each representing a frame.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @classmethod
    def run(
        cls,
        preprocessors: list[typing.Self],
        tile: csi_tiles.Tile,
        images: list[np.ndarray],
        output_path: str = None,
        log_queue: logging.handlers.QueueHandler = None,
    ) -> list[np.ndarray]:
        """
        Runs as many preprocessors as desired on the frame images.
        :param preprocessors: a list of TilePreprocessor objects.
        :param tile: the tile to run the preprocessor on.
        :param images: a list of np.ndarrays, each representing a frame.
        :param output_path: a str representing the path to save outputs.
        :param log_queue: a logging.handlers.QueueHandler object for logging.
        :return: a list of np.ndarrays, each representing a frame.
        """
        if isinstance(preprocessors, TilePreprocessor):
            preprocessors = [preprocessors]
        # Run through the preprocessors
        for p in preprocessors:
            start_time = time.time()

            # Prepare logging
            if log_queue is not None:
                p.log.addHandler(log_queue)

            new_images = None

            # Populate the anticipated file paths for saving if needed
            if output_path is not None and p.save:
                file_paths = [
                    os.path.join(output_path, p.__repr__(), frame.get_file_name())
                    for frame in csi_frames.Frame.get_frames(tile)
                ]
                # Check if the preprocessor outputs already exist; load if so
                if all([os.path.exists(file_path) for file_path in file_paths]):
                    new_images = [
                        cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        for file_path in file_paths
                    ]
                    p.log.debug(f"Loaded previously saved output for tile {tile.n}")
            else:
                file_paths = None

            if new_images is None:
                # We couldn't load anything; run the preprocessor
                new_images = p.preprocess(images)
                dt = f"{time.time() - start_time:.3f} sec"
                p.log.debug(f"Preprocessed tile {tile.n} in {dt}")
            if file_paths is not None:
                # Save if desired
                for file_path, image in zip(file_paths, new_images):
                    cv2.imwrite(file_path, image)
                p.log.debug(f"Saved images for tile {tile.n}")

            # Update the images
            images = new_images
        return images


class TileSegmenter(ABC):
    """
    Abstract class for a tile segmenter.
    """

    @abstractmethod
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
        # List of output mask types that this segmenter can output; must exist
        self.mask_types = [mask_type for mask_type in MaskType]

    @abstractmethod
    def segment(self, frame_images: list[np.ndarray]) -> dict[MaskType, np.ndarray]:
        """
        Segments the frames of a tile to enumerated mask(s), not modifying frame_images.
        Mask(s) should be returned in a dict with labeled types.
        :param frame_images: a list of np.ndarrays, each representing a frame.
        :return: a dict of np.ndarrays, each representing a mask.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @classmethod
    def run(
        cls,
        segmenters: list[typing.Self],
        tile: csi_tiles.Tile,
        images: list[np.ndarray],
        output_path: str = None,
        log_queue: logging.handlers.QueueHandler = None,
    ) -> dict[MaskType, np.ndarray]:
        """
        Runs as many segmenters as desired on the frame images.
        :param segmenters: a list of TileSegmenter objects.
        :param tile: the tile to run the segmenter on.
        :param images: a list of np.ndarrays, each representing a frame.
        :param output_path: a str representing the path to save outputs.
        :param log_queue: a logging.handlers.QueueHandler object for logging.
        :return: a dict of np.ndarrays, each representing a mask.
        """
        if isinstance(segmenters, TileSegmenter):
            segmenters = [segmenters]
        # Run through the segmenters
        masks = {}
        for s in segmenters:
            start_time = time.time()

            # Prepare logging
            if log_queue is not None:
                s.log.addHandler(log_queue)

            new_masks = None

            # Populate the anticipated file paths for saving if needed
            if output_path is not None and s.save:
                file_paths = {
                    key: os.path.join(
                        output_path, s.__repr__(), f"{tile.n}-{key.value}.tif"
                    )
                    for key in MaskType
                }
                # Check if the segmenter outputs already exist; load if so
                if all([os.path.exists(file_paths[key]) for key in file_paths]):
                    new_masks = {
                        key: cv2.imread(file_paths[key], cv2.IMREAD_UNCHANGED)
                        for key in file_paths
                    }
                    s.log.debug(f"Loaded previously saved output for tile {tile.n}")
            else:
                file_paths = None

            if new_masks is None:
                # We couldn't load anything; run the segmenter
                new_masks = s.segment(images)
                dt = f"{time.time() - start_time:.3f} sec"
                s.log.debug(f"Segmented tile {tile.n} in {dt}")

            if file_paths is not None:
                # Save if desired
                for key, file_path in file_paths.items():
                    cv2.imwrite(file_path, new_masks[key])
                s.log.debug(f"Saved masks for tile {tile.n}")

            # Update the masks
            for key in new_masks:
                if key in masks:
                    s.log.warning(f"{key} mask has already been populated; ignoring")
                else:
                    masks[key] = new_masks[key]
        return masks


class ImageFilter(ABC):
    """
    Abstract class for an image-based event filter.
    """

    @abstractmethod
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
        # List of output mask types that this filter can output; must exist
        # Expected to include EVENT, REMOVED
        self.mask_types = [mask_type for mask_type in MaskType]

    @abstractmethod
    def filter_images(
        self,
        frame_images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
    ) -> dict[MaskType, np.ndarray]:
        """
        Using frame_images and masks, returns new masks that should have filtered out
        unwanted objects from the existing masks.
        Should not be in-place, i.e. should not modify frame_images or masks.
        Returns a dict of masks that will overwrite the existing masks on identical keys.
        :param frame_images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :return: a dict of np.ndarrays, each representing a mask; now filtered.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @classmethod
    def run(
        cls,
        image_filters: list[typing.Self],
        tile: csi_tiles.Tile,
        images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
        output_path: str = None,
        log_queue: logging.handlers.QueueHandler = None,
    ) -> dict[MaskType, np.ndarray]:
        """
        Runs as many image filters as desired on the frame images.
        :param image_filters: a list of ImageFilter objects.
        :param tile: the tile to run the image filter on.
        :param images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :param output_path: a str representing the path to save outputs.
        :param log_queue: a logging.handlers.QueueHandler object for logging.
        :return: a dict of np.ndarrays, each representing a mask.
        """
        if isinstance(image_filters, ImageFilter):
            image_filters = [image_filters]
        # Run through the image filters
        for f in image_filters:
            start_time = time.time()

            # Prepare logging
            if log_queue is not None:
                f.log.addHandler(log_queue)

            new_masks = None

            # Populate the anticipated file paths for saving if needed
            if output_path is not None and f.save:
                file_paths = {
                    key: os.path.join(
                        output_path, f.__repr__(), f"{tile.n}-{key.value}.tif"
                    )
                    for key in MaskType
                }
                # Check if the image filter outputs already exist; load if so
                if all([os.path.exists(file_paths[key]) for key in file_paths]):
                    new_masks = {
                        key: cv2.imread(file_paths[key], cv2.IMREAD_UNCHANGED)
                        for key in file_paths
                    }
                    f.log.debug(f"Loaded previously saved output for tile {tile.n}")
            else:
                file_paths = None

            if new_masks is None:
                # We couldn't load anything; run the image filter
                new_masks = f.filter_images(images, masks)
                dt = f"{time.time() - start_time:.3f} sec"
                f.log.debug(f"Filtered tile {tile.n} in {dt}")

            if file_paths is not None:
                # Save if desired
                for key, file_path in file_paths.items():
                    cv2.imwrite(file_path, new_masks[key])
                f.log.debug(f"Saved masks for tile {tile.n}")

            # Update the masks
            masks.update(new_masks)
        return masks


class FeatureExtractor(ABC):
    """
    Abstract class for a feature extractor.
    """

    @abstractmethod
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
        # Must have a list of column names for the features it will populate
        self.columns = []

    @abstractmethod
    def extract_features(
        self,
        frame_images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
        events: csi_events.EventArray,
    ) -> pd.DataFrame:
        """
        Using frame_images, masks, and events, returns new features as a pd.DataFrame.
        :param frame_images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :param events: an EventArray, potentially with populated feature data.
        :return: a pd.DataFrame representing new feature data for events.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @classmethod
    def run(
        cls,
        feature_extractors: list[typing.Self],
        tile: csi_tiles.Tile,
        images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
        events: csi_events.EventArray,
        output_path: str = None,
        log_queue: logging.handlers.QueueHandler = None,
    ) -> csi_events.EventArray:
        """
        Runs as many feature extractors as desired on the frame images.
        :param feature_extractors: a list of FeatureExtractor objects.
        :param tile: the tile to run the feature extractor on.
        :param images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :param events: an EventArray without feature data.
        :param output_path: a str representing the path to save outputs.
        :param log_queue: a logging.handlers.QueueHandler object for logging.
        :return: an EventArray with populated feature data.
        """
        if isinstance(feature_extractors, FeatureExtractor):
            feature_extractors = [feature_extractors]
        # Run through the feature extractors
        for e in feature_extractors:
            start_time = time.time()

            # Prepare logging
            if log_queue is not None:
                e.log.addHandler(log_queue)

            new_features = None

            # Populate the anticipated file paths for saving if needed
            if output_path is not None and e.save:
                file_path = os.path.join(output_path, e.__repr__(), f"{tile.n}.parquet")
                # Check if the feature extractor outputs already exist; load if so
                if os.path.exists(file_path):
                    new_features = pd.read_parquet(file_path)
                    e.log.debug(f"Loaded previously saved output for tile {tile.n}")
            else:
                file_path = None

            if new_features is None:
                # We couldn't load anything; run the feature extractor
                new_features = e.extract_features(images, masks, events)
                dt = f"{time.time() - start_time:.3f} sec"
                e.log.debug(f"Extracted features for tile {tile.n} in {dt}")

            # TODO: handle column name collisions
            # Maybe checks beforehand? Maybe drops columns here?

            if file_path is not None:
                # Save if desired
                new_features.to_parquet(file_path, index=False)
                e.log.debug(f"Saved features for tile {tile.n}")

            # Update the features
            events.add_features(new_features)
        return events


class FeatureFilter(ABC):
    """
    Abstract class for a feature-based event filter.
    """

    @abstractmethod
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

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @classmethod
    def run(
        cls,
        feature_filters: list[typing.Self],
        metadata: csi_scans.Scan | csi_tiles.Tile,
        events: csi_events.EventArray,
        output_path: str = None,
        log_queue: logging.handlers.QueueHandler = None,
    ) -> tuple[csi_events.EventArray, csi_events.EventArray]:
        """
        Runs as many feature filters as desired on the event features.
        :param feature_filters: a list of FeatureFilter objects.
        :param metadata: the scan or tile to run the feature filter on.
        :param events: an EventArray with populated feature data.
        :param output_path: a str representing the path to save outputs.
        :param log_queue: a logging.handlers.QueueHandler object for logging.
        :return: two EventArrays: tuple[remaining, filtered]
        """
        if isinstance(feature_filters, FeatureFilter):
            feature_filters = [feature_filters]
        all_filtered = []
        # Run through the feature filters
        for f in feature_filters:
            start_time = time.time()

            # Slightly different handling for scans and tiles
            if isinstance(metadata, csi_scans.Scan):
                file_stub = f"{f.__repr__()}"
                log_msg = f"all of {metadata.slide_id}"
            elif isinstance(metadata, csi_tiles.Tile):
                file_stub = f"{f.__repr__()}/{metadata.n}"
                log_msg = f"tile {metadata.n}"
            else:
                raise ValueError("metadata must be a Scan or Tile object")

            # Prepare logging
            if log_queue is not None:
                f.log.addHandler(log_queue)

            remaining_events = None
            filtered_events = None

            # Populate the anticipated file paths for saving if needed
            if output_path is not None and f.save:
                file_paths = [
                    os.path.join(output_path, f"{file_stub}-remaining.h5"),
                    os.path.join(output_path, f"{file_stub}-filtered.h5"),
                ]
                # Check if the feature filter outputs already exist; load if so
                if all([os.path.exists(file_path) for file_path in file_paths]):
                    remaining_events, filtered_events = [
                        csi_events.EventArray.load_hdf5(file_path)
                        for file_path in file_paths
                    ]
                    f.log.debug(f"Loaded previously saved events for {log_msg}")
            else:
                file_paths = None

            if remaining_events is None:
                # We couldn't load anything; run the feature filter
                remaining_events, filtered_events = f.filter_features(events)
                dt = f"{time.time() - start_time:.3f} sec"
                f.log.debug(f"Filtered for {log_msg} in {dt}")

            if file_paths is not None:
                # Save if desired
                remaining_events.to_hdf5(file_paths[0])
                filtered_events.to_hdf5(file_paths[1])
                f.log.debug(f"Saved events for {log_msg}")

            # Update events
            all_filtered.append(filtered_events)
            events = remaining_events
        # Combine filtered events
        all_filtered = csi_events.EventArray.from_list(all_filtered)
        return events, all_filtered


class EventClassifier(ABC):
    """
    Abstract class for an event classifier.
    """

    @abstractmethod
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
        # Must have a list of column names for the metadata it will populate
        self.columns = []

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

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version}"

    @classmethod
    def run(
        cls,
        event_classifiers: list[typing.Self],
        metadata: csi_scans.Scan | csi_tiles.Tile,
        events: csi_events.EventArray,
        output_path: str = None,
        log_queue: logging.handlers.QueueHandler = None,
    ):
        """
        Runs as many event classifiers as desired on the event features.
        :param event_classifiers: a list of EventClassifier objects.
        :param metadata: the scan or tile to run the feature filter on.
        :param events: an EventArray with potentially populated metadata.
        :param output_path: a str representing the path to save outputs.
        :param log_queue: a logging.handlers.QueueHandler object for logging.
        :return: an EventArray with populated metadata.
        """
        if isinstance(event_classifiers, EventClassifier):
            event_classifiers = [event_classifiers]
        # Run through the event classifiers
        for c in event_classifiers:
            start_time = time.time()

            # Slightly different handling for scans and tiles
            if isinstance(metadata, csi_scans.Scan):
                file_name = f"{c.__repr__()}"
                log_msg = f"all of {metadata.slide_id}"
            elif isinstance(metadata, csi_tiles.Tile):
                file_name = f"{c.__repr__()}/{metadata.n}"
                log_msg = f"tile {metadata.n}"
            else:
                raise ValueError("metadata must be a Scan or Tile object")

            # Prepare logging
            if log_queue is not None:
                c.log.addHandler(log_queue)

            new_events = None

            # Populate the anticipated file paths for saving if needed
            if output_path is not None and c.save:
                file_path = os.path.join(output_path, f"{file_name}.h5")
                # Check if the event classifier outputs already exist; load if so
                if os.path.exists(file_path):
                    new_events = csi_events.EventArray.load_hdf5(file_path)
                    c.log.debug(f"Loaded previously saved output for {log_msg}")
            else:
                file_path = None

            if new_events is None:
                # We couldn't load anything; run the event classifier
                new_events = c.classify_events(events)
                dt = f"{time.time() - start_time:.3f} sec"
                c.log.debug(f"Classified events for {log_msg} in {dt}")

            # TODO: handle column name collisions
            # Maybe checks beforehand? Maybe drops columns here?

            if file_path is not None:
                # Save if desired
                new_events.to_hdf5(file_path)
                c.log.debug(f"Saved events for {log_msg}")

            # Update events
            events = new_events
        return events


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
        os.makedirs(self.output_path, exist_ok=True)
        self.save_steps = save_steps
        if self.save_steps:
            os.makedirs(os.path.join(self.output_path, "temp"), exist_ok=True)
        self.max_workers = max_workers
        self.verbose = verbose
        # Set up logger
        self.log = csi_logging.get_logger(
            level=logging.DEBUG if self.verbose else logging.INFO,
            log_file=os.path.join(self.output_path, "pipeline.log"),
        )
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

        # Log queue for multiprocessing
        self.log_queue = None

    def run(self) -> csi_events.EventArray:
        """
        Runs the pipeline on the scan.
        """

        start_time = time.time()
        self.log.info("Beginning to run the pipeline on the scan...")

        # Get all tiles
        tiles = csi_tiles.Tile.get_tiles(self.scan)
        # First, do tile-specific steps
        max_workers = min(multiprocessing.cpu_count() - 1, 61)
        # Don't need to parallelize; probably for debugging
        if self.max_workers <= 1:
            events = list(map(run_tile_pipeline, itertools.repeat(self), tiles))
        else:
            # Run in parallel with multiprocess logging
            self.log_queue = csi_logging.start_multiprocess_logging(
                self.log, using_ProcessPoolExecutor=True
            )
            with ProcessPoolExecutor(max_workers) as executor:
                events = list(
                    executor.map(run_tile_pipeline, itertools.repeat(self), tiles)
                )
            csi_logging.stop_multiprocess_logging()

        # Combine EventArrays from all tiles
        events = csi_events.EventArray.from_list(events)

        # Prepare path for intermediate (module-by-module) outputs
        if self.save_steps:
            temp_output_path = os.path.join(self.output_path, "temp")
        else:
            temp_output_path = None

        # Filter events by features at the scan level
        events, filtered = FeatureFilter.run(
            self.scan_feature_filters, self.scan, events, temp_output_path
        )
        # TODO: something with filtered

        events = EventClassifier.run(
            self.scan_event_classifiers, self.scan, events, temp_output_path
        )

        self.log.info(f"Pipeline finished in {(time.time() - start_time)/60:.2f} min")

        return events


def run_tile_pipeline(
    pipeline: ScanPipeline,
    tile: csi_tiles.Tile,
    output_path: str = None,
):
    """
    Runs tile-specific pipeline steps on a tile.
    :param pipeline:
    :param tile: the tile to run the pipeline on.
    :param output_path: a str representing the path to save outputs.
    :return: a csi_events.EventArray with populated features and potentially
             populated metadata.
    """
    # Set up multiprocess logging on the client side
    log = csi_logging.get_logger(
        name="csi_tile_pipeline",
        level=logging.DEBUG if pipeline.verbose else logging.INFO,
        queue=pipeline.log_queue,
    )
    # Prepare queue handler for logging
    if pipeline.log_queue is not None:
        queue_handler = logging.handlers.QueueHandler(pipeline.log_queue)
    else:
        queue_handler = None
    # Prepare output path for intermediate saving
    if pipeline.save_steps:
        output_path = os.path.join(pipeline.output_path, "temp")
    else:
        output_path = None

    # Load the tile frames
    frames = csi_frames.Frame.get_frames(tile)
    images = [frame.get_image()[0] for frame in frames]
    log.debug(f"Loaded {len(images)} frame images for tile {tile.n}")

    images = TilePreprocessor.run(
        pipeline.preprocessors, tile, images, queue_handler, output_path
    )

    masks = TileSegmenter.run(
        pipeline.segmenters, tile, images, queue_handler, output_path
    )

    masks = ImageFilter.run(
        pipeline.image_filters, tile, images, masks, queue_handler, output_path
    )

    events = event_extracter.mask_to_events(pipeline.scan, tile, masks[MaskType.EVENT])

    events = FeatureExtractor.run(
        pipeline.feature_extractors,
        tile,
        images,
        masks,
        events,
        queue_handler,
        output_path,
    )

    events, filtered = FeatureFilter.run(
        pipeline.tile_feature_filters, tile, events, queue_handler, output_path
    )
    # TODO: save combined filtered somewhere, probably up above
    # Only if there is more than one filter

    events = EventClassifier.run(
        pipeline.tile_event_classifiers, tile, events, queue_handler, output_path
    )
    return events
