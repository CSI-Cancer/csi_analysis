import sys
import os
import cv2
import h5py
import gc
import argparse
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from skimage import feature, filters, measure, segmentation
from slideutils.utils import utils
from slideutils.utils.frame import Frame
import normalization_utils as hist_utils

from torch.utils.data import DataLoader
from wbc_dataloader import CustomImageDataset
from wbc_classifier import CNNModel

from cellpose import models

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import tqdm

basic_features_dtypes = {
    'frame_id': 'uint16', 'cell_id': 'uint16', 'y': 'uint16', 'x': 'uint16',
    'area': 'uint32', 'eccentricity': 'float16', 'DAPI_mean': 'uint16',
    'TRITC_mean': 'uint16', 'CY5_mean': 'uint16', 'FITC_mean': 'uint16'
}

def preprocess_frame(frame, params):
    """
    Preprocess frame images for segmentation and feature extraction.
    """
    # Apply tophat
    if params["tophat_size"] != 0:
        tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params["tophat_size"], params["tophat_size"])
        )

        for ch in params["mask_ch"]:
            i = frame.get_ch(ch)
            frame.image[..., i] = cv2.morphologyEx(
                frame.image[..., i], cv2.MORPH_TOPHAT, tophat_kernel
            )

def segment_frame(args):
    """
    Segment a single frame using the assigned Cellpose model.
    """
    frame, cp_model, params = args
    rgb = utils.channels_to_bgr(frame.image, [0, 3], [2, 3], [1, 3])
    mask, _, _ = cp_model.eval(
        rgb, diameter=15, channels=[0, 0], batch_size=8
    )

    return mask
    

def extract_features(frame, params):
    """
    Extract features from a segmented frame.
    """

    # Calculating basic features
    features = frame.calc_basic_features()


    # Initializing variables to hold event image and mask crops
    images = None
    masks = None

    # Filtering events
    if len(params["filters"]) != 0:
        features = utils.filter_events(
            features, params["filters"], params["verbose"]
        )


    # Extracting event images
    if features is not None:
        if params["extract_img"]:
            images, masks = frame.extract_crops(
                features, params["width"], mask_flag=params["mask_flag"]
            )
    

    return {"features": features, "images": images, "masks": masks}

def read_and_preprocess_frame(frame_id, in_path, channels, starts, name_format, include_edge, params):
    """
    Read and preprocess a single frame.
    """
    # Collect image names
    paths = utils.generate_tile_paths(
        path=in_path,
        frame_id=frame_id,
        starts=starts,
        name_format=name_format,
    )

    # Instantiate Frame object
    frame = Frame(frame_id=frame_id, channels=channels, paths=paths)

    # Discard edge frames
    if not include_edge and frame.is_edge():
        return None

    frame.readImage()
    # Preprocess frame
    preprocess_frame(frame, params)
    return frame

def post_process_frame(frame, params):
    """
    Post-process a frame after segmentation.
    """
    image = hist_utils.match_image_to_reference(
        frame, params, params['normalize']
    )

    return image

def process_frames(args):

    mp.set_start_method('spawn', force=True)

    # Create logger object
    logger = utils.get_logger(__name__, args.verbose)

    # Input variables
    in_path = args.input
    output = args.output
    n_frames = args.nframes
    channels = args.channels
    starts = args.starts
    offset = args.offset
    name_format = args.format
    n_threads = args.threads
    include_edge = args.include_edge_frames

    # Segmentation parameters
    params = {
        "tophat_size": args.tophat_size,
        "opening_size": args.open_size,
        #"blur_size": args.blur_size,
        #"blur_sigma": args.blur_sigma,
        #"thresh_size": args.thresh_size,
        #"thresh_offset": args.thresh_offsets,
        #"min_dist": args.min_seed_dist,
        #"seed_ch": args.seed_channel,
        "mask_ch": args.mask_channels,
        "mask_path": args.mask_path,
        "name_format": args.format,
        "exclude_border": args.exclude_border,
        "filters": args.filter,
        "extract_img": args.extract_images,
        "width": args.width,
        "mask_flag": args.mask_flag,
        "verbose": args.verbose,
        "normalize": args.normalize,
        "channel_names": args.channels,
        "debug": args.debug,
    }

    logger.info("Detecting available GPUs...")
    n_gpus = torch.cuda.device_count()
    #logger.info(f"Number of GPUs available: {n_gpus}")

    if n_gpus == 0 and args.device != 'cpu':
        logger.error("No GPUs detected. Exiting.")
        sys.exit(-1)
    #add a check if the device is within the range of available GPUs
    if args.device != 'cpu':
        if args.device not in [f'cuda:{i}' for i in range(n_gpus)] and args.device != 'mps' and args.device != 'cuda':
            logger.error("The device specified is not within the range of available GPUs. Exiting.")
            sys.exit(-1)
    
    logger.info("Loading Cellpose model to GPU...")
    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=args.mask_model,device=torch.device(args.device))
    logger.info("Finished loading Cellpose model.")

    logger.info("Loading classifier model...")
    classifier = CNNModel()
    classifier.load_state_dict(torch.load(args.classifier_model))
    classifier.eval()
    logger.info("Finished loading classifier model.")

    logger.info("Loading frames...")
    # Check if there is a selection of frames to process
    if args.selected_frames:
        frame_ids = args.selected_frames
    else:
        frame_ids = [i + offset + 1 for i in range(n_frames)]

    # Read and preprocess frames in parallel
    n_proc = n_threads if n_threads > 0 else mp.cpu_count()

    read_preprocess_partial = partial(
        read_and_preprocess_frame,
        in_path=in_path,
        channels=channels,
        starts=starts,
        name_format=name_format,
        include_edge=include_edge,
        params=params
    )

    with mp.Pool(n_proc) as pool:
        frames = pool.map(read_preprocess_partial, frame_ids)

    # Filter out None frames (edge frames)
    frames = [frame for frame in frames if frame is not None]

    if params['tophat_size'] != 0:
        logger.info("Applying TopHat filter to frames...")

    logger.info("Finished loading and preprocessing frames.")


    logger.info("Segmenting frames...")


    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        masks = list(tqdm.tqdm(executor.map(segment_frame, [(frame, cellpose_model, params) for frame in frames])))

    # Removing edge events
    for i, mask in enumerate(masks):
        labels = np.unique(np.concatenate([mask[0,:], mask[-1,:], mask[:,0], mask[:,-1]]))
        for label in labels:
            masks[i][mask == label] = 0

    for frame, mask in zip(frames, masks):
        frame.mask = mask.astype("uint16")
        # Saving the mask
        if params["mask_path"] is not None:
            frame.writeMask(params["mask_path"])

    del masks, cellpose_model
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("Finished segmenting frames.")

    if args.normalize is not None:
        # Post-process frames using multiprocessing pool
        logger.info("Post-processing frames...")
        post_process_partial = partial(post_process_frame, params=params)

        with mp.Pool(n_proc) as pool:
            norm_images = list(pool.map(post_process_partial, frames))
        
        for frame, n_image in zip(frames, norm_images):
            frame.image = n_image

        del norm_images

        logger.info("Finished post-processing frames.")

    all_features = []
    if args.debug:
        all_images = []
        all_masks = []

    while frames:
        print(f"{len(frames)} frames remaining...")

        chunk, frames = frames[:args.frame_batch], frames[args.frame_batch:]
        
        logger.info("Processing the frames...")
        n_proc = 8

        # Use a context manager to ensure the pool is closed
        with mp.Pool(n_proc) as pool:
            data = pool.map(partial(extract_features, params=params), chunk)


        logger.info("Finished processing the frames.")

        del chunk
        gc.collect()

        logger.info("Collecting features...")
        features = [
            out["features"] for out in data if out["features"] is not None
        ]
        if len(features)==0:
            logger.error("No events to report in this set of frames!")
            sys.exit(-1)
        else:
            features = pd.concat(features, ignore_index=True)
        
        images = None
        masks = None

        if args.extract_images:
            logger.info("Collecting event images...")
            images = np.concatenate(
                [out["images"] for out in data if out["images"] is not None],
                axis=0
            )

            if args.mask_flag:
                logger.info("Collecting event masks...")
                masks = np.concatenate(
                    [out["masks"] for out in data if out["masks"] is not None],
                    axis=0
                )
        
        del data
        gc.collect()
        
        # Applying the input sortings
        if len(args.sort) != 0:
            logger.info("Sorting events...")
            features = utils.sort_events(features, args.sort, args.verbose)
            logger.info("Finished sorting events.")
            images = images[list(features.index)]
            masks = masks[list(features.index)]
            features.reset_index(drop=True, inplace=True)
    
        # Run the classifier on the images
        logger.info("Running classifier on images...")
        #create a dataset and dataloader
        dataset = CustomImageDataset(images, masks, np.zeros(images.shape[0]))
        dataloader = DataLoader(dataset, batch_size=5000, shuffle=False)
        
        all_preds = []

        #run the classifier
        with torch.no_grad():
            classifier.to(args.device)
            classifier.eval()
            for inputs, _ in dataloader:
                inputs = inputs.to(args.device)
                outputs = classifier(inputs) #returns 2 logits, one for each class
                _, preds = torch.max(outputs, 1) #get the class with the highest probability
                #probs = torch.nn.functional.softmax(outputs, dim=1) #get the probabilities
                #probs = probs.cpu().detach().numpy()
                preds = preds.cpu().numpy()
                all_preds.append(preds)
        
        
            # Concatenate all predictions into a single array
        all_preds_concat = np.concatenate(all_preds)

        # Obtain indices where predictions are 0
        indices = np.where(all_preds_concat == 0)[0]  # Extract the array from the tuple

        #subset an additional 45 images, at random, where the classifier predicted 1
        indices_ones = np.where(all_preds_concat == 1)[0]
        indices_ones = np.random.choice(indices_ones, 45, replace=False)
        indices = np.concatenate((indices, indices_ones))

        # Sort the indices
        indices = np.sort(indices)

        # Subset the images and masks based on the predictions
        images = images[indices]
        masks = masks[indices]

        # Subset the features DataFrame based on the predictions
        features = features.iloc[indices, :]
        #reset the index
        features.reset_index(drop=True, inplace=True)

        if args.debug:
            all_images.append(images)
            all_masks.append(masks)

        all_features.append(features)
        
        del dataset, dataloader, images, masks, features
        gc.collect()

    all_features = pd.concat(all_features, axis=0)
    all_features.astype(basic_features_dtypes)

    logger.info("Finished processing all frames.")
    logger.info("Saving data...")
    all_features.to_parquet(output, compression='gzip')

    if args.debug:
        debug_filename = f"{os.path.dirname(output)}/{os.path.basename(output).split('.')[0]}.hdf5"
        with h5py.File(debug_filename, mode='w') as hf:
            hf.create_dataset("images", data=np.concatenate(all_images, axis=0))
            hf.create_dataset("channels", data=args.channels)
            if args.mask_flag:
                hf.create_dataset("masks", data=np.concatenate(all_masks, axis=0))
        all_features.to_hdf(debug_filename, mode='a', key='features')

    logger.info("Finished saving features.")


def main():

    # main inputs
    parser = argparse.ArgumentParser(
        description="Process slide images to identify cells.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input path to slide images",
    )

    parser.add_argument(
        "-o", "--output", type=str, required=True, help="output path"
    )

    parser.add_argument(
        "-m",
        "--mask_path",
        type=str,
        default=None,
        help="mask path to save frame masks if needed",
    )

    parser.add_argument(
        "-f", "--offset", type=int, default=0, help="start frame offset"
    )

    parser.add_argument(
        "-n", "--nframes", type=int, default=2304, help="number of frames"
    )

    parser.add_argument(
        "-c",
        "--channels",
        type=str,
        nargs="+",
        default=["DAPI", "TRITC", "CY5", "FITC"],
        help="channel names",
    )

    parser.add_argument(
        "-s",
        "--starts",
        type=int,
        nargs="+",
        default=[1, 2305, 4609, 9217],
        help="channel start indices",
    )

    parser.add_argument(
        "-F",
        "--format",
        type=str,
        nargs="+",
        default=["Tile%06d.tif"],
        help="image name format",
    )

    parser.add_argument(
        "--mask_model",
        required=True,
        type=str,
        help="path to model for segmentation model",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity level"
    )

    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="number of threads for parallel processing",
    )

    # Segmentation parameters
    parser.add_argument(
        "--tophat_size", type=int, default=0, help="TopHat filter kernel size"
    )

    parser.add_argument(
        "--open_size",
        type=int,
        default=5,
        help="Open morphological filter kernel size",
    )

    parser.add_argument(
        "--mask_channels",
        type=str,
        nargs="+",
        default=["DAPI", "TRITC", "CY5", "FITC"],
        help="channels to segment",
    )

    parser.add_argument(
        "--exclude_border",
        default=False,
        action="store_true",
        help="exclude events that are on image borders",
    )

    parser.add_argument(
        "--include_edge_frames",
        default=False,
        action="store_true",
        help="include frames that are on the edge of slide",
    )

    parser.add_argument(
        "--selected_frames",
        type=int,
        nargs="*",
        default=[],
        help="list of selected frames to be processed",
    )

    parser.add_argument(
        "--extract_images",
        default=True,
        action="store_true",
        help="extract images of detected events for inference [always true]",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=75,
        help="""
        size of the event images to be cropped from slide images [always 75]
        """,
    )

    parser.add_argument(
        "--frame_batch",
        type=int,
        default=200,
        help="frame processing batch size",
    )

    parser.add_argument(
        "--mask_flag",
        default=True,
        action="store_true",
        help="store event masks when extracting images [always True]",
    )

    parser.add_argument(
        "--sort",
        type=str,
        nargs=2,
        action="append",
        default=[],
        help="""
        sort events based on feature values.

        Usage:   <command> --sort <feature> <order>
        Example: <command> --sort TRITC_mean I
        order:    I: Increasing / D: Decreasing
        """,
    )

    parser.add_argument(
        "--filter",
        type=str,
        nargs=3,
        action="append",
        default=[],
        help="""
        feature range for filtering detected events.

        Usage:    <command> --feature_range <feature> <min> <max>
        Example:  <command> --feature_range DAPI_mean 0 10000
        """,
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="activate debug mode to save hdf5 files",
    )

    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="input path to h5 reference file for normalization",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to use for segmentation and inference. set to cpu if not GPU available. set to cuda:0, cuda:1, etc. for specific GPU. set to mps for mac.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="number of workers for cellpose segmentation. Limited with memory, so monitor your usage.",
    )

    parser.add_argument(
        "--classifier_model",
        type=str,
        default=None,
        help="path to model for classification",
    )

    args = parser.parse_args()

    # Check if channel names and channel indices have same length
    if len(args.channels) != len(args.starts) and len(args.channels) != len(args.format):
        print("number of channels do not match with number of starts or name formats")
        sys.exit(-1)

    process_frames(args)

if __name__ == "__main__":
    main()
