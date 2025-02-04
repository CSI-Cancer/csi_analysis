import os
from typing import Sequence

from loguru import logger


import functools
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from csi_images.csi_scans import Scan
from csi_images.csi_events import EventArray
from csi_images import csi_images

from csi_analysis.training.channel_classifier.channel_classifier.modeling import model
from csi_analysis.pipelines.scan import MaskType, FeatureExtractor


class ChannelClassifier(FeatureExtractor):
    CHANNELS = {
        "DAPI": (0, 0, 1),
        "AF647": (0, 1, 0),
        "AF555": (1, 0, 0),
        "AF488": (1, 1, 1),
    }

    IMAGE_SIZE = 50

    def __init__(
        self,
        scan: Scan,
        model_path: str,
        device: torch.device = torch.device("cpu"),
        channels: Sequence[str | int] = ("AF555", "AF647", "DAPI", "AF488"),
        colors: Sequence[tuple[float, float, float]] = (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
        ),
        batch_size: int = 256,
    ):
        """
        Creates a ChannelClassifier object with custom configuration for
        classifying channels of events in images.
        :param scan:
        :param model_path:
        :param device:
        :param batch_size:
        """
        self.scan = scan
        self.channels = channels
        self.colors = colors
        if len(self.channels) != len(self.colors):
            raise ValueError("channels and colors must have the same length")
        if isinstance(self.channels[0], str):
            # Convert channel string names to indices
            self.channels = scan.get_channel_indices(self.channels)

        # Load the model
        self.model_path = model_path
        self.device = device
        model_state = torch.load(model_path, map_location=device, weights_only=True)
        # Figured out num_layers_per_block is 6 from the model definition
        self.model = model.GenericCNN().to(device)
        # For some reason, the model is saved with "module." in front of the keys
        # for key in list(model_state["model_state_dict"].keys()):
        #     if key.startswith("module."):
        #         new_key = key.replace("module.", "")
        #         model_state["model_state_dict"][new_key] = model_state[
        #             "model_state_dict"
        #         ].pop(key)
        self.model.load_state_dict(model_state["model_state_dict"])
        self.model.eval()

        self.batch_size = batch_size

    def __repr__(self):
        return f"{self.__class__.__name__}-{os.path.basename(self.model_path)})"

    def extract_features(
        self,
        events: EventArray,
        images: list[np.ndarray] | list[list[np.ndarray]],
        masks: list[np.ndarray] | list[dict[MaskType, np.ndarray]] = None,
    ) -> EventArray:
        # Set up the RGB creation function with color arguments
        colors = [(0, 0, 0)] * len(self.scan.channels)
        for i, color in zip(self.channels, self.colors):
            colors[i] = color
        make_rgb = functools.partial(csi_images.make_rgb, colors=colors)

        # Create a dataset and dataloader
        dataset = RGBImageDataset(make_rgb, images, masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        confidence = []
        classification = []
        logger.info("Classifying channels in batches...")
        with torch.no_grad():
            for i, (x, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
                x = x.to(self.device)
                # Get the classification probabilities
                probs = torch.softmax(self.model(x), dim=1)
                # Get the classification labels and confidence
                prob, index = torch.max(probs, dim=1)
                # Append to appropriate lists
                confidence.append(prob.detach().cpu())
                classification.append(index.detach().cpu())
        torch.cuda.empty_cache()
        classification = [model.CLASSES[i] for i in classification]
        events.add_metadata(
            pd.DataFrame(
                {
                    "channel_classification": classification,
                    "channel_confidence": confidence,
                }
            )
        )
        return events


class RGBImageDataset(Dataset):

    def __init__(
        self,
        make_rgb: callable,
        images: list[list[np.ndarray]],
        masks: list[np.ndarray] | list[dict[MaskType, np.ndarray]] = None,
        labels: torch.ByteTensor = None,
        image_size: int = ChannelClassifier.IMAGE_SIZE,
        transform=None,
    ):
        self.make_rgb = make_rgb
        self.images = images
        self.masks = masks
        if labels is None:
            labels = torch.zeros(len(images), dtype=torch.uint8)
        self.labels = labels
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        images = self.images[i]
        # Make into (H, W, C) RGB image with range [0, 1]
        image = self.make_rgb(images)
        image = csi_images.scale_bit_depth(image, np.float32)
        # Crop the image to the center
        x_start = (image.shape[1] - self.image_size) // 2
        y_start = (image.shape[0] - self.image_size) // 2
        image = image[
            y_start : y_start + self.image_size, x_start : x_start + self.image_size, :
        ]

        if self.masks is not None:
            mask = self.masks[i]
            mask = torch.tensor(mask, dtype=torch.bool)
            image = image * mask

        if self.transform is not None:
            # Also transforms it to a (C, H, W) tensor
            image = self.transform(image)

        return image, self.labels[i]
