#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:09:35 2024

@author: vishn
"""
import numpy as np
import pandas as pd
import operator

from csi_images import csi_scans
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, erosion
from skimage.measure import find_contours, regionprops, regionprops_table
from skimage.segmentation import watershed
from skimage.util import img_as_float
from skimage import filters
from skimage.filters import sobel
from skimage.segmentation import relabel_sequential

from csi_analysis.pipelines.scan_pipeline import TileSegmenter, MaskType


class ClassicSegmenter(TileSegmenter):
    def __init__(self, scan: csi_scans.Scan, version: str, save_output: bool = False):
        pass

    def __repr__(self):
        pass

    def segment(self, frame_images: list[np.ndarray]) -> dict[MaskType, np.ndarray]:
        pass


def thresh_sauvola(image, w=5, k=0.2):
    # Calculate Sauvola threshold
    sauvola_threshold = filters.threshold_sauvola(image, window_size=(2 * w + 1), k=k)

    # Apply threshold
    binary_image = image > sauvola_threshold

    return binary_image


def normalize_with_median(image):
    """Median-based normalization"""
    # Create a copy of the image to avoid changing the original image
    image_copy = np.copy(image)

    image_median = np.median(image_copy)
    image_min = image_copy.min()
    image_max = image_copy.max()

    image_copy[image_copy <= image_median] = (
        image_copy[image_copy <= image_median] - image_min
    )
    image_copy[image_copy > image_median] = (
        (image_copy[image_copy > image_median] - image_median)
        / (image_max - image_median)
    ) + image_median

    return image_copy


def remove_objects_size(
    labeled_image, max_size, operation_str, default_image, reenumerate=False
):
    # Map operator strings to actual operator functions
    operations = {
        "==": operator.eq,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
    }

    # Get the actual operator function from the map
    operation = operations.get(operation_str)

    if operation is None:
        raise ValueError(
            "Invalid operation string. Choose from '==', '>', '<', '>=', '<='."
        )

    # Get region properties for both labeled_image and default_image
    labeled_regions = regionprops(labeled_image)
    default_regions = regionprops(default_image)

    # Initialize a mask of zeros with the same shape as the labeled_image
    remove_mask = np.zeros_like(labeled_image, dtype=bool)
    count = 0

    # Iterate over each region in the default_image and labeled_image, comparing areas
    for i, default_region in enumerate(default_regions):
        if i < len(labeled_regions):  # Ensure index exists in labeled_regions
            labeled_region = labeled_regions[i]

            # Compare the region's area in the default_image with the max_size
            if operation(default_region.area, max_size):
                # Update the mask for the current region in the labeled_image
                remove_mask[labeled_image == labeled_region.label] = True
                count += 1

    # Apply the mask to the labeled_image to set the pixels of specified objects to 0
    labeled_image[remove_mask] = 0

    # Re-enumeration if requested
    if reenumerate:
        labeled_image, _, _ = relabel_sequential(labeled_image)

    return labeled_image


def rmObjects_id(x, index, reenumerate=True):
    # Ensure x is a NumPy array
    x = np.array(x)

    # Create a mask for all values to be removed
    mask = np.isin(x, index)
    x[mask] = 0  # Set the identified positions to 0

    return x


def propagate(x, seeds, mask=None, lambda_value=1e-4):
    """
    Voronoi-based segmentation of an image using seeds.

    Parameters:
    - x: An array representing the image to segment.
    - seeds: An array containing the seeds for segmentation.
    - mask: An optional binary mask for regions to segment. If None, the whole image is considered.
    - lambda_value: Regularization parameter controlling the trade-off between Euclidean distance and gradient contribution.

    Returns:
    - An array containing the labeled segmented regions.
    """
    # Convert image and seeds to float
    image = img_as_float(x)

    binary_mask = np.where(mask, 1, 0).astype(np.uint8)

    gradient = sobel(image)

    segmented_image = watershed(gradient, markers=seeds, mask=binary_mask)

    return segmented_image


def process_mask(image, min_area_size, indvimgsize, cellsmask_copy):
    # adaptive thresh
    mask = thresh_sauvola(image, 50, -3.5)
    mask = erosion(mask, disk(2.8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask, _, _ = relabel_sequential(mask)

    mask = remove_objects_size(mask, min_area_size, "<", cellsmask_copy)
    mask = remove_objects_size(mask, indvimgsize, ">=", cellsmask_copy)
    return mask


def ocontour(x):
    contours = find_contours(x, 1)
    return [np.fliplr(c) for c in contours]


def splitObjects(x):
    labeled_array, _, _ = relabel_sequential(x)
    regions = regionprops(labeled_array)
    objects = [region.coords for region in regions]
    return objects


def gblob(x0=15, n=49, alpha=0.8, beta=1.2):
    xx = np.linspace(-x0, x0, n)
    xx = np.sqrt(xx[:, None] ** 2 + xx[None, :] ** 2)
    z = np.exp(-(xx**2) / (2 * alpha**2)) - 0.65 * np.exp(-(xx**2) / (2 * beta**2))
    return z / np.sum(z)


def compute_features(
    x,
    ref,
    methods_noref=["computeFeatures_moment", "computeFeatures_shape"],
    methods_ref=["computeFeatures_basic", "computeFeatures_moment"],
    xname="x",
    refnames=None,
    properties=False,
    expandRef=None,
    **kwargs,
):

    def computeFeatures_basic(
        x, ref, properties=False, basic_quantiles=[0.01, 0.05, 0.5, 0.95, 0.99], xs=None
    ):
        qnames = [f'b_q{str(q).replace(".", "")}' for q in basic_quantiles]

        if not properties:
            x = checkx(x)
            if xs is None:
                xs = splitObjects(x)
            if len(xs) == 0:
                return None
            ref = convertRef(ref)["a"]

            # Compute features
            features = []
            for obj in xs:
                z = ref[obj[:, 0], obj[:, 1]]
                q = np.quantile(z, basic_quantiles)
                feature = {
                    "b_mean": np.mean(z),
                    "b_sd": np.std(z),
                    "b_mad": np.median(np.abs(z - np.median(z))),
                }
                feature.update(dict(zip(qnames, q)))
                features.append(feature)

            features_df = pd.DataFrame(features)

            # Special processing for single points
            single_point = [len(obj) == 1 for obj in xs]
            features_df.loc[single_point, "b_sd"] = 0

            return features_df
        else:
            # Feature properties
            properties_df = pd.DataFrame(
                {
                    "name": ["b_mean", "b_sd", "b_mad"] + qnames,
                    "translation_invariant": True,
                    "rotation_invariant": True,
                }
            )
            return properties_df

    def computeFeatures_shape(x, properties=False, xs=None):
        if properties:
            # Return feature properties
            properties_df = pd.DataFrame(
                {
                    "name": [
                        "s_area",
                        "s_perimeter",
                        "s_radius_mean",
                        "s_radius_sd",
                        "s_radius_min",
                        "s_radius_max",
                    ],
                    "translation_invariant": [True, True, True, True, True, True],
                    "rotation_invariant": [True, True, True, True, True, True],
                }
            )
            return properties_df

        x = checkx(x)

        # Get labeled array (ignore the number of features)
        labeled_array, _, _ = relabel_sequential(x)

        # Pass only the labeled array to regionprops
        regions = regionprops(labeled_array)

        if len(regions) == 0:
            return None

        # Compute features for each region
        features = []
        for region in regions:
            area = region.area
            min_row, min_col, max_row, max_col = region.bbox
            region_image = region.image

            # Find contours on the region_image
            contours = find_contours(region_image, level=0.5)
            if contours:
                # Use the longest contour (outer contour)
                contour = max(contours, key=len)
                if len(contour) > 0:
                    # Adjust contour coordinates to the global image coordinates
                    contour[:, 0] += min_row
                    contour[:, 1] += min_col

                    centroid = np.array(region.centroid)[::-1]  # Reverse to (x, y)
                    radius = np.linalg.norm(contour - centroid, axis=1)
                    radius_mean = np.mean(radius)
                    radius_sd = np.std(radius)
                    radius_min = np.min(radius)
                    radius_max = np.max(radius)

                    # Compute perimeter as the sum of distances between consecutive contour points
                    contour_closed = np.vstack([contour, contour[0]])
                    diffs = np.diff(contour_closed, axis=0)
                    distances = np.hypot(diffs[:, 0], diffs[:, 1])
                    perimeter = np.sum(distances)

                    features.append(
                        {
                            "s_area": area,
                            "s_perimeter": perimeter,
                            "s_radius_mean": radius_mean,
                            "s_radius_sd": radius_sd,
                            "s_radius_min": radius_min,
                            "s_radius_max": radius_max,
                        }
                    )

        # Convert features list to DataFrame
        features_df = pd.DataFrame(features)
        return features_df

    def computeFeatures_moment(x, ref=None, properties=False, xs=None):
        if properties:
            # Return feature properties
            properties_df = pd.DataFrame(
                {
                    "name": [
                        "m_cx",
                        "m_cy",
                        "m_majoraxis",
                        "m_eccentricity",
                        "m_theta",
                    ],
                    "translation_invariant": [False, False, True, True, True],
                    "rotation_invariant": [True, True, True, True, False],
                }
            )
            return properties_df

        x = checkx(x)
        if xs is None:
            xs = splitObjects(x)
        if len(xs) == 0:
            return None
        if ref is None:
            ref = np.ones_like(x)
        ref = convertRef(ref)["a"]

        # Create a labeled image from xs
        labeled_array = np.zeros_like(x, dtype=int)
        for i, coords in enumerate(xs, start=1):
            labeled_array[coords[:, 0], coords[:, 1]] = i

        # Use regionprops_table to compute properties
        props = regionprops_table(
            labeled_array,
            intensity_image=ref,
            properties=[
                "label",
                "weighted_centroid",
                "orientation",
                "major_axis_length",
                "eccentricity",
            ],
        )

        # Convert to DataFrame
        features = pd.DataFrame(props)

        # Rename columns to match desired output
        features.rename(
            columns={
                "weighted_centroid-0": "m_cx",
                "weighted_centroid-1": "m_cy",
                "major_axis_length": "m_majoraxis",
                "eccentricity": "m_eccentricity",
                "orientation": "m_theta",
            },
            inplace=True,
        )

        # Handle NaNs and Infs
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Select columns to match desired output
        columns = ["m_cx", "m_cy", "m_majoraxis", "m_eccentricity", "m_theta"]
        features = features[columns]

        return features

    # Map method names to functions internally
    method_mapping = {
        "computeFeatures_moment": computeFeatures_moment,
        "computeFeatures_shape": computeFeatures_shape,
        "computeFeatures_basic": computeFeatures_basic,
    }

    # Main function logic
    x = checkx(x)
    ref = convertRef(ref, refnames)
    refnames = list(ref.keys())
    nref = len(ref)

    if expandRef:
        ref = expandRef(ref, refnames)
        refnames = list(ref.keys())
        nref = len(ref)

    if not properties:
        xs = splitObjects(x)
        if not xs:
            return None

        # Extract features without reference
        features_noref = pd.concat(
            [
                method_mapping[method](x=x, properties=False, xs=xs, **kwargs)
                for method in methods_noref
            ],
            axis=1,
        )

        # Extract features with reference
        features_ref = []
        for i in range(nref):
            ref_features = pd.concat(
                [
                    method_mapping[method](
                        x=x, ref=ref[refnames[i]], properties=False, xs=xs, **kwargs
                    )
                    for method in methods_ref
                ],
                axis=1,
            )
            features_ref.append(ref_features)

        features_ref = {refnames[i]: features_ref[i] for i in range(nref)}
        features = {"0": features_noref}
        features.update(features_ref)

        # Rename columns with prefixes
        for key in features:
            features[key].columns = [f"{key}_{col}" for col in features[key].columns]

        features_df = pd.concat(features.values(), axis=1)
        features_df.columns = [f"{xname}_{col}" for col in features_df.columns]
        return features_df

    else:
        # Extract property features without reference
        pfeatures_noref = pd.concat(
            [
                method_mapping[method](properties=True, **kwargs)
                for method in methods_noref
            ],
            axis=1,
        )

        # Extract property features with reference
        pfeatures_ref = pd.concat(
            [
                method_mapping[method](properties=True, **kwargs)
                for method in methods_ref
            ],
            axis=1,
        )
        pfeatures_ref = {refnames[i]: pfeatures_ref for i in range(nref)}

        pfeatures = {"0": pfeatures_noref}
        pfeatures.update(pfeatures_ref)

        # Rename columns for property features
        for key in pfeatures:
            pfeatures[key] = pd.DataFrame(
                pfeatures[key],
                columns=["name", "translation_invariant", "rotation_invariant"],
            )
            pfeatures[key]["name"] = f"{key}_" + pfeatures[key]["name"]

        pfeatures_df = pd.concat(pfeatures.values())
        pfeatures_df["name"] = f"{xname}_" + pfeatures_df["name"]
        return pfeatures_df


def stack_objects(x, ref, combine=True, bg_col=(0, 0, 0), ext=None):
    # Ensure x is grayscale
    if len(x.shape) != 2:
        raise ValueError("'x' must be a grayscale image")

    # Ensure ref is compatible
    if x.shape != ref.shape[:2]:
        raise ValueError("'x' and 'ref' must have the same spatial dimensions")

    nobj = int(np.max(x))  # Number of objects (maximum label in x)
    nbChannels = (
        ref.shape[2] if len(ref.shape) > 2 else 1
    )  # Number of color channels in ref

    cropped_objects = []

    for obj_idx in range(1, nobj + 1):
        obj_mask = x == obj_idx
        obj_coords = np.argwhere(obj_mask)

        xl, xr = obj_coords[:, 0].min(), obj_coords[:, 0].max()
        yl, yr = obj_coords[:, 1].min(), obj_coords[:, 1].max()

        cropped_ref = ref[xl : xr + 1, yl : yr + 1]

        if nbChannels == 1:
            cropped_ref = cropped_ref[:, :, np.newaxis]

        cropped_objects.append(cropped_ref)

    if combine:
        return np.array(
            cropped_objects, dtype=object
        )  # Return as a numpy array of objects
    else:
        return cropped_objects  # Return as a list


def checkx(x):
    if not isinstance(x, np.ndarray):
        raise ValueError("'x' must be a 2D array")
    if x.ndim == 3 and x.shape[2] > 1:
        raise ValueError("'x' must be a 2D array")
    if x.ndim == 3:
        x = x[:, :, 0]
    return x


def splitObjects(x):
    labeled_array, _, _ = relabel_sequential(x)
    regions = regionprops(labeled_array)
    objects = [region.coords for region in regions]
    return objects


def convertRef(ref, refnames=None):
    if isinstance(ref, np.ndarray):
        if ref.ndim == 2:
            ref = [ref]
        elif ref.ndim == 3:
            ref = [ref[:, :, i] for i in range(ref.shape[2])]
        else:
            raise ValueError("'ref' must be a 2D or 3D array")
    elif isinstance(ref, list):
        ref = [r if r.ndim == 2 else r[:, :, 0] for r in ref]
    else:
        raise ValueError(
            "'ref' must be an array or a list containing the reference images"
        )

    if refnames is None:
        refnames = [chr(i) for i in range(97, 97 + len(ref))]
    if len(refnames) != len(ref):
        raise ValueError("'refnames' must have the same length as 'ref'")

    return {refnames[i]: ref[i] for i in range(len(ref))}
