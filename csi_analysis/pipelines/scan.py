import os
import time
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from enum import Enum
from abc import ABC, abstractmethod
from loguru import logger

try:
    import imageio.v3 as imageio
except ImportError:
    # Not required for implementing abstract classes
    imageio = None

import numpy as np
import pandas as pd

from tqdm import tqdm
import functools
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from csi_images.csi_scans import Scan
from csi_images.csi_tiles import Tile
from csi_images.csi_frames import Frame
from csi_images.csi_events import EventArray


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
    def preprocess(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Preprocess the frames of a tile, preferably in-place.
        Should return the frames in the same order.
        No coordinate system changes should occur here, as they are handled elsewhere.
        :param images: a list of np.ndarrays, each representing a frame.
        :return: a list of np.ndarrays, each representing a frame.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        :return: a file-system safe string representing the module and key options.
        """
        return f"{self.__class__.__name__}"

    def run(
        self,
        tile: Tile,
        images: list[np.ndarray],
        output_path: str = None,
    ) -> list[np.ndarray]:
        """
        Runs the preprocessor on one tile's images.
        Consider overriding this method to run on many or all tiles.
        :param tile: the tile to run the preprocessor on.
        :param images: a list of np.ndarrays, each representing a frame's image.
        :param output_path: a str representing the path to save outputs.
        Will add an appropriately named folder to this path.
        :return: a list of np.ndarrays, each representing a frame.
        """
        start_time = time.time()
        new_images = None

        # Populate the anticipated file paths for saving if needed
        if imageio is None and output_path is not None:
            output_path = None
            logger.warning("imageio is required for saving outputs; skipping save")
        elif output_path is not None:
            # Add a folder for this module's outputs
            output_path = os.path.join(output_path, self.__repr__())
            os.makedirs(output_path, exist_ok=True)
            # Get the paths for the frames
            file_paths = [f.get_file_path(output_path) for f in Frame.get_frames(tile)]
            # Check if the outputs already exist; load if so
            if all([os.path.exists(file_path) for file_path in file_paths]):
                new_images = [imageio.imread(file_path) for file_path in file_paths]
                logger.debug(f"Loaded saved output for tile {tile.n}")

        if new_images is None:
            # We couldn't load anything; run the preprocessor
            new_images = self.preprocess(images)
            dt = f"{time.time() - start_time:.3f} sec"
            logger.debug(f"Preprocessed tile {tile.n} in {dt}")

        # Save if desired
        if output_path is not None:
            for file_path, image in zip(file_paths, new_images):
                imageio.imwrite(file_path, image, compression="deflate")
            logger.debug(f"Saved preprocessed images for tile {tile.n}")
        return new_images


class TileSegmenter(ABC):
    """
    Abstract class for a tile segmenter.
    """

    mask_type: MaskType  #: The type of mask that this segmenter outputs.

    @abstractmethod
    def segment(
        self, images: list[np.ndarray], masks: dict[MaskType, np.ndarray]
    ) -> dict[MaskType, np.ndarray]:
        """
        Segments the frames of a tile to enumerated mask(s), not modifying images.
        Mask(s) should be returned in a dict with labeled types.
        :param images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :return: masks, but with an additional entry for the new mask type OR
        overwritten entry for the same mask type.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        :return: a file-system safe string representing the module and key options.
        """
        return f"{self.__class__.__name__}"

    def run(
        self,
        tile: Tile,
        images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray] = None,
        output_path: str = None,
    ) -> dict[MaskType, np.ndarray]:
        """
        Runs the segmenter on one tile's images.
        Consider overriding this method to run on many or all tiles.
        :param tile: the tile to run the segmenter on.
        :param images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :param output_path: a str representing the path to save outputs.
        :return: masks, but with an additional entry for the new mask type.
        """
        start_time = time.time()
        new_mask = None

        # Check on masks
        if masks is None:
            masks = {}
        if self.mask_type in masks:
            # Throw a warning that we will end up overwriting the mask
            logger.warning(f"{self.mask_type} mask already exists; overwriting")

        # Attempt to load the mask from a previous run
        if imageio is None and output_path is not None:
            output_path = None
            logger.warning("imageio is required for saving outputs; skipping save")
        elif output_path is not None:
            # Add a folder for this module's outputs
            output_path = os.path.join(output_path, self.__repr__())
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, f"{tile.n}.tif")
            # Check if the outputs already exist; load if so
            if os.path.exists(output_path):
                new_mask = imageio.imread(output_path)
                logger.debug(f"Loaded saved output for tile {tile.n}")

        if new_mask is None:
            # We couldn't load anything; run the segmenter
            masks = self.segment(images, masks)
            dt = f"{time.time() - start_time:.3f} sec"
            logger.debug(f"Segmented tile {tile.n} in {dt}")
        else:
            # Loaded a mask, update it in the dict
            masks[self.mask_type] = new_mask

        # Save if desired
        if output_path is not None:
            imageio.imwrite(output_path, masks[self.mask_type], compression="deflate")
            logger.debug(f"Saved masks for tile {tile.n}")

        return masks


class ImageFilter(ABC):
    """
    Abstract class for an image-based event filter.
    """

    mask_type: MaskType  #: The type of mask that this filter overwrites.

    @abstractmethod
    def filter_images(
        self, images: list[np.ndarray], masks: dict[MaskType, np.ndarray]
    ) -> dict[MaskType, np.ndarray]:
        """
        Using images and masks, returns a modified masks that should have
        filtered out unwanted objects from the existing mask at mask_type.
        :param images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :return: a dict of np.ndarrays, each representing a mask, with a modified
        mask for the specified mask_type.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        :return: a file-system safe string representing the module and key options.
        """
        return f"{self.__class__.__name__}"

    def run(
        self,
        tile: Tile,
        images: list[np.ndarray],
        masks: dict[MaskType, np.ndarray],
        output_path: str = None,
    ) -> dict[MaskType, np.ndarray]:
        """
        Removes elements from a mask.
        :param tile: the tile to run the image filter on.
        :param images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :param output_path: a str representing the path to save outputs.
        :return: a dict of np.ndarrays, each representing a mask.
        """
        start_time = time.time()
        new_mask = None

        if self.mask_type not in masks:
            raise ValueError(
                f"Mask type {self.mask_type} not found in masks; "
                f"cannot filter out masks that don't exist"
            )

        # Attempt to load the mask from a previous run
        if imageio is None and output_path is not None:
            output_path = None
            logger.warning("imageio is required for saving outputs; skipping save")
        if output_path is not None:
            # Add a folder for this module's outputs
            output_path = os.path.join(output_path, self.__repr__())
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, f"{tile.n}.tif")
            # Check if the outputs already exist; load if so
            if os.path.exists(output_path):
                new_mask = imageio.imread(output_path)
                logger.debug(f"Loaded saved output for tile {tile.n}")

        if new_mask is None:
            # We couldn't load anything; run the image filter
            masks = self.filter_images(images, masks)
            dt = f"{time.time() - start_time:.3f} sec"
            logger.debug(f"Filtered tile {tile.n} in {dt}")
        else:
            # Loaded a mask, update it in the dict
            masks[self.mask_type] = new_mask

        # Save if desired
        if output_path is not None:
            imageio.imwrite(output_path, masks[self.mask_type], compression="deflate")
            logger.debug(f"Saved masks for tile {tile.n}")

        return masks


class FeatureExtractor(ABC):
    """
    Abstract class for a feature extractor.
    """

    @abstractmethod
    def extract_features(
        self,
        events: EventArray,
        images: list[np.ndarray] | list[list[np.ndarray]],
        masks: dict[MaskType, np.ndarray] | list[dict[MaskType, np.ndarray]],
    ) -> EventArray:
        """
        Using images, masks, and events, adds new features to events.
        :param events: an EventArray, potentially with populated feature data.
        :param images: a list of np.ndarrays, each representing a frame; or a list thereof.
        :param masks: a dict of np.ndarrays, each representing a mask; or a list thereof.
        :return: an EventArray with new populated feature data.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        :return: a file-system safe string representing the module and key options.
        """
        return f"{self.__class__.__name__}"

    def run(
        self,
        target: Tile | Scan,
        events: EventArray,
        images: list[np.ndarray] | list[list[np.ndarray]],
        masks: dict[MaskType, np.ndarray],
        output_path: str = None,
    ) -> EventArray:
        """
        Runs the feature extractor on a tile's images. Consider overriding this
        method to run on many or all tiles.
        :param target: the scan or tile to run the feature extractor on.
        :param events: an EventArray.
        :param images: a list of np.ndarrays, each representing a frame.
        :param masks: a dict of np.ndarrays, each representing a mask.
        :param output_path: a str representing the path to save outputs.
        :return: an EventArray with more feature data.
        """
        start_time = time.time()

        # Slightly different handling for scans and tiles
        if isinstance(target, Tile):
            tag = f"tile {target.n}"
            file = f"{target.n}"
        elif isinstance(target, Scan):
            tag = f"all of {target.slide_id}"
            file = f"{target.slide_id}"
        else:
            raise ValueError("metadata must be a Tile or Scan object")

        # Attempt to load the feature-filled events from a previous run
        if output_path is not None:
            # Add a folder for this module's outputs
            output_path = os.path.join(output_path, self.__repr__())
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, file)
            # Check if the outputs already exist; load if so
            if os.path.exists(output_path):
                events = EventArray.load_hdf5(output_path)
                logger.debug(f"Loaded saved output for {tag}")
                return events  # Exit early

        # Didn't exit early
        events = self.extract_features(events, images, masks)
        dt = f"{time.time() - start_time:.3f} sec"
        logger.debug(f"Extracted features for {tag} in {dt}")

        # Save if desired
        if output_path is not None:
            events.save_hdf5(output_path)
            logger.debug(f"Saved features for {tag}")

        return events


class FeatureFilter(ABC):
    """
    Abstract class for a feature-based event filter.
    """

    @abstractmethod
    def filter_features(self, events: EventArray) -> tuple[EventArray, EventArray]:
        """
        Removes events from an event array based on feature values.
        :param events: a EventArray with populated features.
        :return: two EventArray objects: tuple[remaining, filtered]
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        :return: a file-system safe string representing the module and key options.
        """
        return f"{self.__class__.__name__}"

    def run(
        self,
        target: Tile | Scan,
        events: EventArray,
        output_path: str = None,
    ) -> tuple[EventArray, EventArray]:
        """
        Runs as many feature filters as desired on the event features.
        :param target: the scan or tile to run the feature filter on.
        :param events: an EventArray with populated feature data.
        :param output_path: a str representing the path to save outputs.
        :return: two EventArrays: tuple[remaining, filtered]
        """
        start_time = time.time()
        remaining = None
        filtered = None

        # Slightly different handling for scans and tiles
        if isinstance(target, Tile):
            tag = f"tile {target.n}"
            file = f"{target.n}"
        elif isinstance(target, Scan):
            tag = f"all of {target.slide_id}"
            file = f"{target.slide_id}"
        else:
            raise ValueError("metadata must be a Tile or Scan object")

        # Attempt to load the results from a previous run
        if output_path is not None:
            # Add a folder for this module's outputs
            output_path = os.path.join(output_path, self.__repr__())
            os.makedirs(output_path, exist_ok=True)
            # Get the paths for the results
            file_paths = [
                os.path.join(output_path, f"{file}-remaining.h5"),
                os.path.join(output_path, f"{file}-filtered.h5"),
            ]
            # Check if the outputs already exist; load if so
            if all([os.path.exists(file_path) for file_path in file_paths]):
                remaining, filtered = [EventArray.load_hdf5(f) for f in file_paths]
                logger.debug(f"Loaded saved output for {tag}")

        if remaining is None or filtered is None:
            # We couldn't load anything; run the feature filter
            remaining, filtered = self.filter_features(events)
            dt = f"{time.time() - start_time:.3f} sec"
            logger.debug(f"Filtered for {tag} in {dt}")

        # Save if desired
        if output_path is not None:
            remaining.save_hdf5(file_paths[0])
            filtered.save_hdf5(file_paths[1])
            logger.debug(f"Saved events for {tag}")

        return remaining, filtered


class EventClassifier(ABC):
    """
    Abstract class for an event classifier.
    """

    @abstractmethod
    def classify_events(self, events: EventArray) -> EventArray:
        """
        Classifies events based on features, then populates the metadata.
        :param events: a EventArray with populated features.
        :return: a EventArray with populated metadata.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        :return: a file-system safe string representing the module and key options.
        """
        return f"{self.__class__.__name__}"

    def run(
        self,
        target: Scan | Tile,
        events: EventArray,
        output_path: str = None,
    ):
        """
        Runs the event classifier on the event features.
        :param target: the scan or tile to run the feature filter on.
        :param events: an EventArray with potentially populated metadata.
        :param output_path: a str representing the path to save outputs.
        :return: an EventArray with populated metadata.
        """
        start_time = time.time()

        # Slightly different handling for scans and tiles
        if isinstance(target, Tile):
            tag = f"tile {target.n}"
            file = f"{target.n}"
        elif isinstance(target, Scan):
            tag = f"all of {target.slide_id}"
            file = f"{target.slide_id}"
        else:
            raise ValueError("metadata must be a Tile or Scan object")

        # Attempt to load the results from a previous run
        if output_path is not None:
            # Add a folder for this module's outputs
            output_path = os.path.join(output_path, self.__repr__())
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, f"{file}.h5")
            # Check if the outputs already exist; load if so
            if os.path.exists(output_path):
                events = EventArray.load_hdf5(output_path)
                logger.debug(f"Loaded saved output for {tag}")
                return events  # Exit early

        # Didn't exit early
        events = self.classify_events(events)
        dt = f"{time.time() - start_time:.3f} sec"
        logger.debug(f"Classified events for {tag} in {dt}")

        # Save if desired
        if output_path is not None:
            events.save_hdf5(output_path)
            logger.debug(f"Saved events for {tag}")

        return events


class ReportGenerator(ABC):
    """
    Abstract class for a report generator.
    """

    @abstractmethod
    def make_report(
        self,
        events: EventArray,
        output_path: str,
    ) -> bool:
        """
        Creates a report based off of the passed events. Unlike other modules,
        the outputs may vary greatly. This method should be used to generate
        a report in the desired format and should check on the outputs to ensure
        that the report was generated successfully.
        :param output_path: a str representing the path to save outputs.
        :param events: a EventArray with populated features.
        :return: True for success.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        :return: a file-system safe string representing the module and key options.
        """
        return f"{self.__class__.__name__}"


class ScanPipeline:
    """
    This is an **example pipeline** for processing a scan. It assumes that
    particular modules are meant to be run on tiles vs. scans. You may need to
    write a similar class for your own pipeline, depending on the modules you use.

    For instance, GPU-heavy modules like model-based feature extraction may be
    run serially rather than in parallel due to GPU memory load.

    Here, we assume that tiles of the scan cannot be stitched together, nor is
    it desired to do so. Instead, we perform image tasks on the tiles separately.

    However, we do assume that events from different tiles can be stitched together and
    analyzed as a whole, so we allow for event filtering and classification at both
    the tile and scan levels.

    This has the bonus of never fully loading the scan into memory; while all
    events are loaded into memory at the end, this is much less memory-intensive
    (e.g. 2.5e6 events at 1 KB each is 2.5 GB, compared to ~20GB of image data).
    """

    def __init__(
        self,
        scan: Scan,
        output_path: str,
        preprocessors: list[TilePreprocessor] = (),
        segmenters: list[TileSegmenter] = (),
        image_filters: list[ImageFilter] = (),
        feature_extractors: list[FeatureExtractor] = (),
        tile_feature_filters: list[FeatureFilter] = (),
        tile_event_classifiers: list[EventClassifier] = (),
        scan_feature_filters: list[FeatureFilter] = (),
        scan_event_classifiers: list[EventClassifier] = (),
        report_generators: list[ReportGenerator] = (),
        excluded_border_size: int = 0,
        save_steps: bool = False,
        max_workers: int = 61,
        log_options: dict = None,
    ):
        # Set up loguru logger
        self.log_options = log_options
        if log_options is not None and len(log_options) > 0:
            logger.remove(0)
            for sink, options in log_options.items():
                logger.add(sink, **options)
        self.scan = scan
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.save_steps = save_steps
        self.excluded_border_size = excluded_border_size
        self.max_workers = max_workers

        self.preprocessors = preprocessors
        self.segmenters = segmenters
        self.image_filters = image_filters
        self.feature_extractors = feature_extractors
        self.tile_feature_filters = tile_feature_filters
        self.tile_event_classifiers = tile_event_classifiers
        self.scan_feature_filters = scan_feature_filters
        self.scan_event_classifiers = scan_event_classifiers
        self.report_generators = report_generators

    def run(self) -> EventArray:
        """
        Runs the pipeline on the scan.
        """

        start_time = time.time()
        logger.info("Beginning to run the pipeline on the scan...")

        # Prepare path for intermediate (module-by-module) outputs
        if self.save_steps:
            temp_path = os.path.join(self.output_path, "temp")
            os.makedirs(temp_path, exist_ok=True)
        else:
            temp_path = None

        # Get tiles, excluding the border
        tiles = Tile.get_tiles_by_xy_bounds(
            self.scan,
            (
                self.excluded_border_size,
                self.excluded_border_size,
                self.scan.roi[0].tile_cols - self.excluded_border_size,
                self.scan.roi[0].tile_rows - self.excluded_border_size,
            ),
        )
        # First, do tile-specific steps
        max_workers = min(multiprocessing.cpu_count() - 1, 61)
        # Don't need to parallelize; probably for debugging
        tile_job = functools.partial(
            process_tile,
            output_path=temp_path,
            preprocessors=self.preprocessors,
            segmenters=self.segmenters,
            image_filters=self.image_filters,
            feature_extractors=self.feature_extractors,
            feature_filters=self.tile_feature_filters,
            event_classifiers=self.tile_event_classifiers,
        )
        if self.max_workers <= 1:
            events = list(tqdm(map(tile_job, tiles)))
        else:
            with ProcessPoolExecutor(
                max_workers, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                events = list(
                    tqdm(
                        executor.map(
                            tile_job, tiles, itertools.repeat(self.log_options)
                        )
                    )
                )

        # Combine EventArrays from all tiles
        events = EventArray.merge(events)

        # Filter events by features at the scan level
        for f in self.scan_feature_filters:
            events, _ = f.run(self.scan, events, temp_path)

        # Classify events at the scan level
        for c in self.scan_event_classifiers:
            events = c.run(self.scan, events, temp_path)

        # Save the final events
        events.save_hdf5(os.path.join(self.output_path, f"{self.scan.slide_id}"))

        # Generate reports
        for r in self.report_generators:
            success = r.make_report(events, self.output_path)
            if not success:
                logger.warning(
                    f"Report generation failed for {r}; see logs for details"
                )

        logger.info(f"Pipeline finished in {(time.time() - start_time)/60:.2f} min")

        return events


def process_tile(
    tile: Tile,
    output_path: str = None,
    preprocessors: list[TilePreprocessor] = (),
    segmenters: list[TileSegmenter] = (),
    image_filters: list[ImageFilter] = (),
    feature_extractors: list[FeatureExtractor] = (),
    feature_filters: list[FeatureFilter] = (),
    event_classifiers: list[EventClassifier] = (),
    log_options: dict = None,
):
    """
    Runs tile-specific pipeline steps on a tile.
    :param tile: the tile to run the modules on.
    :param output_path: a str representing the path to save outputs or None to not save
    :param preprocessors: a list of TilePreprocessor objects.
    :param segmenters: a list of TileSegmenter objects.
    :param image_filters: a list of ImageFilter objects.
    :param feature_extractors: a list of FeatureExtractor objects.
    :param feature_filters: a list of FeatureFilter objects.
    :param event_classifiers: a list of EventClassifier objects.
    :param log_options:
    :return: a EventArray with populated features and potentially
             populated metadata.
    """
    start_time = time.time()

    # Set up multiprocess logging on the client side
    if log_options is not None and len(log_options) > 0:
        logger.remove(0)
        for sink, options in log_options.items():
            logger.add(sink, **options, enqueue=True)

    # Load the tile frames
    frames = Frame.get_frames(tile)
    images = [frame.get_image() for frame in frames]
    logger.debug(f"Loaded {len(images)} frame images for tile {tile.n}")

    for p in preprocessors:
        images = p.run(tile, images, output_path)

    # Multiple segmenters may require some coordination
    masks = {}
    for s in segmenters:
        masks = s.run(tile, images, output_path)

    for f in image_filters:
        masks = f.run(tile, images, masks, output_path)

    # Convert masks to an EventArray
    events = EventArray.from_mask(masks[MaskType.EVENT], tile)

    for e in feature_extractors:
        events = e.run(tile, events, images, masks, output_path)

    for f in feature_filters:
        events, _ = f.run(tile, events, output_path)

    for c in event_classifiers:
        events = c.run(tile, events, output_path)

    logger.info(f"Tile {tile.n} finished in {time.time() - start_time:.3f} sec")
    return events
