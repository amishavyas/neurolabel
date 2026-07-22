from importlib.resources import as_file, files
import numpy as np
import nibabel as nib
from nilearn.image import load_img
import nilearn.datasets


def extract_parcel_mask(parcellation, parcel_num):
    """
    Create a binary mask for a single parcel from a labeled NIfTI parcellation.

    Parameters
    ----------
    parcellation : str, pathlib.Path, or Niimg-like object
        Labeled parcellation image.
    parcel_num : int
        Label of the parcel to extract. Must correspond to a valid
        non-background parcel (background is label 0).

    Returns
    -------
    mask : ndarray
        Binary NumPy array with 1s for voxels belonging to the selected
        parcel and 0s elsewhere.
    mask_img : nibabel.Nifti1Image
        NIfTI image containing the binary mask in the original image space.

    Raises
    ------
    ValueError
        If `parcel_num` is 0 or is not present in the parcellation.
    """
    img = load_img(parcellation)
    data = np.round(img.get_fdata()).astype(int)

    if parcel_num == 0 or parcel_num not in np.unique(data):
        raise ValueError(
            "Invalid parcel number. Please enter a valid integer.")

    mask = (data == parcel_num).astype(np.int32)
    mask_img = nib.Nifti1Image(mask, img.affine, img.header)

    return mask, mask_img


_PARCELLATIONS = {
    "neurosynth50": "Neurosynth_Parcellation_k50.nii.gz",
    "neurosynth100": "Neurosynth_Parcellation_k100.nii.gz",
    "neurosynth200": "Neurosynth_Parcellation_k200.nii.gz",
}


def load_neurosynth(name: str) -> nib.Nifti1Image:
    """
    Load a bundled Neurosynth parcellation.

    Parameters
    ----------
    name
        Name of the parcellation to load.

    Returns
    -------
    nibabel.Nifti1Image
        Loaded parcellation image.

    Raises
    ------
    ValueError
        If ``name`` is not a supported Neurosynth parcellation.
    """
    try:
        filename = _PARCELLATIONS[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_PARCELLATIONS))
        raise ValueError(
            f"Unknown Neurosynth parcellation '{name}'. "
            f"Available parcellations: {available}."
        ) from exc

    resource = files("data.parcellations").joinpath(filename)

    with as_file(resource) as path:
        image = nib.load(path)

        return nib.Nifti1Image(
            np.asanyarray(image.dataobj),
            image.affine,
            image.header.copy(),
        )

_NILEARN_ATLASES = {
    "harvard-oxford": nilearn.datasets.fetch_atlas_harvard_oxford(),
    "schaefer": nilearn.datasets.fetch_atlas_schaefer_2018(),
    "yeo": nilearn.datasets.fetch_atlas_yeo_2011(),
}


def load_nilearn(name: str) -> nib.Nifti1Image:
    """
    Load a bundled Neurosynth parcellation.
    Parameters
    ----------
    name
        Name of the parcellation to load.

    Returns
    -------
    nibabel.Nifti1Image
        Loaded parcellation image.

    Raises
    ------
    ValueError
        If ``name`` is not a supported Nilearn parcellation or parcellation list has not been updated in neurolabel.
    """
    try:
        file_fetcher = _NILEARN_ATLASES[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_NILEARN_ATLASES))
        raise ValueError(
            f"Parcellation '{name}' is unavailable through neurolabel. "
            f"Available parcellations: {available}."
        ) from exc

    return file_fetcher()

def summarize_statistical_map(statistical_map,parcellation="neurosynth200",threshold=None,mode="two-sided",min_voxels=1,):
    """
    Find Neurosynth parcels containing significant voxels.

    Parameters
    ----------
    statistical_map : path or Niimg-like object
        Three-dimensional t-, z-, effect-size, p-value, or already
        thresholded statistical image.

    parcellation : path, Niimg-like object, or str
        Neurosynth parcellation image. The aliases "neurosynth50",
        "neurosynth100", and "neurosynth200" are also accepted.

    threshold : float, optional
        Threshold used to select significant voxels. Not required when
        mode="nonzero".

    mode : {"two-sided", "positive", "negative", "p-value", "nonzero"}
        Thresholding rule:

        two-sided
            abs(value) >= threshold

        positive
            value >= threshold

        negative
            value <= -threshold

        p-value
            value <= threshold

        nonzero
            value != 0

    min_voxels : int
        Minimum significant voxel count needed to report a parcel.

    Returns
    -------
    pandas.DataFrame
        One row per affected parcel.
    """

    valid_modes = {
        "two-sided",
        "positive",
        "negative",
        "p-value",
        "nonzero",
    }

    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {sorted(valid_modes)}."
        )

    if mode != "nonzero" and threshold is None:
        raise ValueError(
            "threshold is required unless mode='nonzero'."
        )

    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be nonnegative.")

    if mode == "p-value" and not 0 <= threshold <= 1:
        raise ValueError(
            "A p-value threshold must be between 0 and 1."
        )

    if min_voxels < 1:
        raise ValueError("min_voxels must be at least 1.")

    stat_img = load_img(statistical_map)

    if (isinstance(parcellation, str) and parcellation.lower() in _PARCELLATIONS):
        parcel_img = load_neurosynth(parcellation)
    else:
        parcel_img = load_img(parcellation)

    if len(stat_img.shape) != 3:
        raise ValueError("statistical_map must be three-dimensional.")

    if len(parcel_img.shape) != 3:
        raise ValueError("parcellation must be three-dimensional.")

    same_grid = (stat_img.shape == parcel_img.shape and np.allclose(stat_img.affine,
            parcel_img.affine,rtol=1e-5,atol=1e-5,))

    if not same_grid:
        # Parcel IDs are categorical, so nearest-neighbor interpolation
        # must be used.
        parcel_img = resample_to_img(parcel_img,stat_img,interpolation="nearest",)

    stat_data = np.asarray(stat_img.get_fdata(),dtype=float,)

    parcel_data = np.round(parcel_img.get_fdata()).astype(int)

    finite = np.isfinite(stat_data)

    if mode == "two-sided":
        significant = (finite & (np.abs(stat_data) >= threshold))

    elif mode == "positive":
        significant = (finite  & (stat_data >= threshold))

    elif mode == "negative":
        significant = (finite & (stat_data <= -threshold))

    elif mode == "p-value":
        significant = (finite & (stat_data <= threshold))

    else:
        significant = (finite & (stat_data != 0))

    total_significant = int(np.count_nonzero(significant))

    # Neurosynth label 0 is background.
    labeled_significant = (significant & (parcel_data > 0))

    total_labeled = int(np.count_nonzero(labeled_significant))

    columns = [
        "parcel_id",
        "significant_voxels",
        "parcel_voxels",
        "percent_of_significant_map",
        "percent_of_parcel_significant",
        "positive_voxels",
        "negative_voxels",
        "mean_value",
        "minimum_value",
        "maximum_value",
        "peak_value",
        "peak_x",
        "peak_y",
        "peak_z",
    ]

    if total_labeled == 0:
        return pd.DataFrame(columns=columns)

    rows = []

    parcel_ids = np.unique(parcel_data[labeled_significant])

    for parcel_id in parcel_ids:
        parcel_id = int(parcel_id)

        parcel_mask = parcel_data == parcel_id

        parcel_significant = (parcel_mask & significant)

        values = stat_data[parcel_significant]
        voxel_indices = np.argwhere(parcel_significant)

        n_significant = int(values.size)

        if n_significant < min_voxels:
            continue

        if mode in {"two-sided", "nonzero"}:
            peak_index = int(np.argmax(np.abs(values)))

        elif mode in {"negative", "p-value"}:
            peak_index = int(np.argmin(values))

        else:
            peak_index = int(np.argmax(values))

        peak_ijk = voxel_indices[peak_index]

        peak_xyz = nib.affines.apply_affine(stat_img.affine,peak_ijk,)

        parcel_voxels = int(np.count_nonzero(parcel_mask))

        rows.append(
            {
                "parcel_id": parcel_id,
                "significant_voxels": n_significant,
                "parcel_voxels": parcel_voxels,
                "percent_of_significant_map": ( 100.0 * n_significant / total_labeled ),
                "percent_of_parcel_significant": ( 100.0 * n_significant / parcel_voxels ),
                "positive_voxels": int(np.count_nonzero(values > 0)),
                "negative_voxels": int(np.count_nonzero(values < 0)),
                "mean_value": float(np.mean(values)),
                "minimum_value": float(np.min(values)),
                "maximum_value": float(np.max(values)),
                "peak_value": float(values[peak_index]),
                "peak_x": float(peak_xyz[0]),
                "peak_y": float(peak_xyz[1]),
                "peak_z": float(peak_xyz[2]),
            }
        )

    result = pd.DataFrame(rows,columns=columns,)

    if not result.empty:
        result = (result.sort_values(["significant_voxels", "parcel_id"],
            ascending=[False, True],).reset_index(drop=True))

    result.attrs["total_significant_voxels"] = (total_significant)

    result.attrs["labeled_significant_voxels"] = (total_labeled)

    result.attrs["significant_voxels_outside_parcellation"] = total_significant - total_labeled

    return result