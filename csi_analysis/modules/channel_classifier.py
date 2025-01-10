import functools

from loguru import logger

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
from csi_analysis.pipelines.scan_pipeline import FeatureExtractor, MaskType


class ChannelClassifier(FeatureExtractor):
    CHANNELS = {
        "DAPI": (0, 0, 1),
        "AF647": (0, 1, 0),
        "AF555": (1, 0, 0),
        "AF488": (1, 1, 1),
    }

    def __init__(
        self,
        scan: Scan,
        model_path: str,
        use_gpu: bool = True,
        batch_size: int = 256,
        save: bool = False,
    ):
        self.scan = scan

        # Determine the device
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                logger.warning("GPU not available, using CPU")
        else:
            device = torch.device("cpu")

        # Load the model
        model_state = torch.load(model_path, map_location=device, weights_only=True)
        # Figured out num_layers_per_block is 6 from the model definition
        self.model = model.DenseNet(num_layers_per_block=6).to(device)
        # For some reason, the model is saved with "module." in front of the keys
        for key in list(model_state["model_state_dict"].keys()):
            if key.startswith("module."):
                new_key = key.replace("module.", "")
                model_state["model_state_dict"][new_key] = model_state[
                    "model_state_dict"
                ].pop(key)
        self.model.load_state_dict(model_state["model_state_dict"])
        self.model.eval()

        self.device = device
        self.batch_size = batch_size
        self.save = save

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.device.__repr__()})"

    def extract_features(
        self,
        events: EventArray,
        images: list[np.ndarray] | list[list[np.ndarray]],
        masks: list[np.ndarray] | list[dict[MaskType, np.ndarray]] = None,
    ) -> EventArray:
        # Set up the RGB creation function with color arguments
        colors = [(0, 0, 0)] * len(self.scan.channels)
        channel_indices = self.scan.get_channel_indices(self.CHANNELS.keys())
        for i, color in zip(channel_indices, self.CHANNELS.values()):
            colors[i] = color
        make_rgb = functools.partial(csi_images.make_rgb, colors=colors)

        # Create a dataset and dataloader
        dataset = RGBImageDataset(make_rgb, images, masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        features = []
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                logger.debug(f"Classifying batch {i} of {len(dataloader)}")
                x = x.to(self.device)
                features.append(self.model(x).detach().cpu())
        torch.cuda.empty_cache()
        features = torch.cat(features)
        # Convert to classification and get the confidence
        _, index = torch.max(features, dim=1)
        classification = [model.CLASSES[i] for i in index.tolist()]
        features = pd.DataFrame(
            {
                "channel_classification": classification,
            }
        )
        events.add_features(features)
        return events


class RGBImageDataset(Dataset):

    def __init__(
        self,
        make_rgb: callable,
        images: list[list[np.ndarray]],
        masks: list[np.ndarray] | list[dict[MaskType, np.ndarray]] = None,
        labels: torch.ByteTensor = None,
        transform=None,
    ):
        self.make_rgb = make_rgb
        self.images = images
        self.masks = masks
        if labels is None:
            labels = torch.zeros(len(images), dtype=torch.uint8)
        self.labels = labels
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
        # Convert the image to RGB, then to a float tensor
        image = self.images[i]
        image = self.make_rgb(image)
        image = csi_images.scale_bit_depth(image, np.float32)

        if self.masks is not None:
            mask = self.masks[i]
            mask = torch.tensor(mask, dtype=torch.bool)
            image = image * mask

        if self.transform is not None:
            # Also transforms it to a (C, H, W) tensor
            image = self.transform(image)

        return image, self.labels[i]
