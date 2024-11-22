"""
Clusters events into groups of maximum size 20, which is the agreed-upon size for
OCULAR-based reporting. Meant to be used in conjunction with ocular_report_montager.py.
"""

from loguru import logger

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from csi_images.csi_events import EventArray

from ..pipelines.scan_pipeline import EventClassifier


class OcularReportClusterer(EventClassifier):
    """
    Creates or replaces columns in event metadata:
    - cluster_id: the cluster id for each event
    """

    COLUMN_NAME = "cluster_id"

    def __init__(
        self,
        columns: list[str] = None,
        max_cluster_size: int = 20,
        sort_by: str = None,
        ascending: bool = True,
        copy: bool = False,
        save: bool = False,
    ):
        self.columns = columns
        self.sort_by = sort_by
        self.max_cluster_size = max_cluster_size
        self.ascending = ascending
        self.copy = copy
        self.save = save

    def __repr__(self):
        return f"{self.__class__.__name__})"

    def classify_events(self, events: EventArray) -> EventArray:
        if self.copy:
            events = events.copy()
        cluster_labels = self.cluster_with_max_size(events)
        events.metadata["cluster_id"] = cluster_labels
        if self.sort_by is not None:
            cluster_mapping = self.sort_clusters(events)
            events.metadata["cluster_id"] = events.metadata["cluster_id"].map(
                cluster_mapping
            )
        return events

    def cluster_with_max_size(self, events: EventArray) -> np.ndarray:
        """
        Clusters events into groups of maximum size 20. Groups that are split to meet the
        maximum size are naively split, not clustered recursively.
        :param events:
        :return: np.ndarray of the cluster labels
        """
        if self.columns is not None:
            columns = self.columns
        else:
            columns = events.features.columns
        # Sub-select the columns to be used for clustering and standardize
        data = events.features[columns]
        data = StandardScaler().fit_transform(data)
        # Starting number of clusters; may change if a cluster is large and must be split
        n_clusters = (len(data) // self.max_cluster_size) + (
            1 if len(data) % self.max_cluster_size > 0 else 0
        )

        # Apply agglomerative clustering
        cluster_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(
            data
        )
        current_max_label = max(cluster_labels)
        unique_cluster_ids = np.unique(cluster_labels)

        # Split them into sub-clusters if above max_size
        for cluster_id in unique_cluster_ids:
            # Filter the DataFrame for the current cluster
            cluster_indices = np.flatnonzero(cluster_labels == cluster_id)
            cluster_size = len(cluster_indices)
            # Skip if the cluster is already small enough
            if cluster_size <= self.max_cluster_size:
                continue

            # Calculate sub-clusters and the size of each sub-cluster
            num_sub_clusters = cluster_size // self.max_cluster_size + (
                1 if cluster_size % self.max_cluster_size > 0 else 0
            )
            sub_cluster_size = cluster_size // num_sub_clusters + (
                1 if cluster_size % self.max_cluster_size > 0 else 0
            )

            # Sub-clusters aside from the first cluster get new cluster labels
            for sub_cluster_id in range(1, num_sub_clusters):
                # Events in the cluster that will be split into a sub-cluster
                start_idx = sub_cluster_id * sub_cluster_size
                end_idx = start_idx + sub_cluster_size

                # Final sub-cluster just fits the remaining events
                if sub_cluster_id == num_sub_clusters - 1:
                    end_idx = cluster_size

                # Assign new cluster ids and increment the current_max_label
                cluster_labels[cluster_indices[start_idx:end_idx]] = (
                    current_max_label + 1
                )
                current_max_label += 1

        logger.debug(f"Clustered {len(events)} events into {n_clusters} clusters")

        return cluster_labels

    def sort_clusters(self, events):
        if self.sort_by is None:
            return events
        # Get the average "interesting" p-value for each cluster
        sort_data = events.get(self.sort_by)
        cluster_means = []
        cluster_ids = pd.unique(events.metadata["cluster_id"])
        for cluster_id in cluster_ids:
            cluster_indices = events.metadata["cluster_id"] == cluster_id
            cluster_means.append(sort_data.loc[cluster_indices].mean()[0])

        # Sort the cluster_ids by their average p-values, descending (reverse=True)
        cluster_order = sorted(
            zip(cluster_means, cluster_ids),
            key=lambda mean_and_id: mean_and_id[0],
            reverse=not self.ascending,
        )

        # Create a dictionary that maps old cluster_id to new, sorted cluster_id
        cluster_mapping = {
            old_index: index for index, (_, old_index) in enumerate(cluster_order)
        }
        return cluster_mapping
