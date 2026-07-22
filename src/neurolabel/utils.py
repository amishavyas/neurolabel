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