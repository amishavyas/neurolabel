import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nibabel import Nifti1Image
from nilearn.image import resample_to_img
from neurolabel.utils import *


def atlas_overlap(target_atlas: str | Nifti1Image, parcel_num: int, label_atlas: str | Nifti1Image):
    """
    Create a dataframe with the labels of regions and number of voxels that overlap across the given parcel number and atlas/parcellation. 

    Parameters:
    nifti_file : str or pathlib.Path
        Path to the labeled parcellation NIfTI file.
    parcel_num : int
        Label of the parcel to extract. Must correspond to a valid
        non-background parcel (background is label 0).
    atlas_string : str
        Name of atlas 
    top_10 : bool
        If True filters to show only the top 10 corresponding parcels. Otherwise, shows all parcels.

    Returns a dataframe containing columns:
        regions = Regional labels for the given atlas/parcellation specified in the atlas_string input, ordered by overlap.
        intersection_voxels = Number of voxels that overlap across the neurosynth parcel and the given atlas/parcellation specified in the atlas_string input.
        parcel_overlap = Percentage of voxels in the neurosynth parcellation that overlap with the given atlas/parcellation specified in the atlas_string input.

    """

    if type(label_atlas) == str:
        atlas = load_nilearn(label_atlas)
    elif type(label_atlas) == nib.Nifti1Image:
        atlas = label_atlas

    _, mask_img = extract_parcel_mask(target_atlas, parcel_num=parcel_num)

    atlas_img = resample_to_img(
        atlas.maps,
        mask_img,
        interpolation="nearest",
    )

    parcel = mask_img.get_fdata().astype(bool)
    atlas_data = atlas_img.get_fdata().astype(int)

    results = []

    for region_id, region_name in enumerate(atlas.labels[1:], start=1):
        region_mask = atlas_data == region_id
        intersection = np.logical_and(parcel, region_mask).sum()

        results.append({
            "regions": region_name,
            "intersection_voxels": intersection,
            "parcel_overlap": intersection / parcel.sum()
        })
    overlap_df = (
        pd.DataFrame(results)
        .sort_values("parcel_overlap", ascending=False)
        .reset_index(drop=True)
    )
    return overlap_df  # [overlap_df["intersection_voxels"] > 0]


def plot_anat(overlap_df, top_n=10, bar_color="#0B5CFF"):
    """
    Plot the top anatomical regions for a parcel as a horizontal bar chart.

    Parameters
    ----------
    decoded_df : pandas.DataFrame
        DataFrame returned by ``atlas_overlap`` containing anatomical overlap
        statistics.
    top_n : int, default=10
        Number of highest-ranked regions to display.
    bar_color : str, default="#0B5CFF"
        Color of the bars.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the horizontal bar chart.
    """
    top = overlap_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        top["regions"],                 # y-axis labels
        top["parcel_overlap"],  # x-axis values
        color=bar_color,
    )
    ax.invert_yaxis()

    ax.set_xlabel("Parcel Overlap", fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    plt.close(fig)

    return fig
