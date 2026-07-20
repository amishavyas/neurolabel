from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from neurolabel.core.extraction import extract_label_mask


def extract_parcel_mask(
    nifti_file: str | Path | nib.spatialimages.SpatialImage | None = None,
    parcel_num: int | None = None,
    *,
    nifti_file_name: str | Path | None = None,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Create a binary mask for an existing discrete image label.

    Parameters
    ----------
    nifti_file
        Path to, or loaded form of, a labeled NIfTI image.
    parcel_num
        Existing integer label, including zero.
    nifti_file_name
        Compatibility keyword for a labeled NIfTI path.

    Returns
    -------
    mask
        Uint8 array with ones at the selected label.
    mask_img
        NIfTI-1 mask preserving the source geometry.
    """
    source = nifti_file_name if nifti_file_name is not None else nifti_file
    if source is None:
        raise ValueError("Provide nifti_file or nifti_file_name.")
    if parcel_num is None:
        raise TypeError("parcel_num must be an integer.")
    return extract_label_mask(source, parcel_num)
