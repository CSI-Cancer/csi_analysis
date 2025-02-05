from typing import Literal

import numpy as np
from csi_images.csi_events import EventArray

from csi_analysis.pipelines.scan import FeatureFilter


class ThresholdingFilter(FeatureFilter):
    def __init__(
        self,
        mode: Literal["mean", "max"] = "max",
        threshold: float = 0.05,
        dtype: np.dtype = np.uint16,
    ):
        self.mode = mode
        if np.issubdtype(dtype, np.unsignedinteger):
            self.threshold = threshold * np.iinfo(dtype).max
        else:
            self.threshold = threshold

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.mode}-{self.threshold})"

    def filter_features(self, events: EventArray) -> tuple[EventArray, EventArray]:
        filtered = []
        filter_columns = [
            c for c in events.features.columns if f"intensity_{self.mode}" in c
        ]
        for column in filter_columns:
            filtered.append(events.rows(events.features[column] < self.threshold))
            events = events.rows(events.features[column] >= self.threshold)
        filtered = EventArray.merge(filtered)
        return events, filtered
