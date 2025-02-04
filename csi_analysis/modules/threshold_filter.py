from typing import Literal

from csi_images.csi_events import EventArray

from csi_analysis.pipelines.scan import FeatureFilter


class ThresholdingFilter(FeatureFilter):
    def __init__(
        self,
        scan: None = None,
        version: Literal["mean", "max"] = "max",
        save: bool = False,
        threshold: int = int(65535 * 0.05),
    ):
        self.version = version
        self.save = save
        self.threshold = threshold

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.threshold})"

    def filter_features(self, events: EventArray) -> tuple[EventArray, EventArray]:
        filtered = []
        filter_columns = [
            c for c in events.features.columns if f"intensity_{self.version}" in c
        ]
        for column in filter_columns:
            filtered.append(events.rows(events.features[column] < self.threshold))
            events = events.rows(events.features[column] >= self.threshold)
        filtered = EventArray.merge(filtered)
        return events, filtered
