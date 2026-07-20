import nibabel as nib
import numpy as np


def extract_parcel_mask(nifti_file=None, nifti_file_name=None, parcel_num=None):
    """
    Create a binary mask for a single parcel from a labeled NIfTI parcellation.

    Parameters
    ----------
    nifti_file : str or pathlib.Path
        Path to the labeled parcellation NIfTI file.
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
    if nifti_file_name:
        img = nib.load(nifti_file_name)
    elif nifti_file:
        img = nifti_file
    else:
        img = None
    data = np.round(img.get_fdata()).astype(int)

    if parcel_num == 0 or parcel_num not in np.unique(data):
        raise ValueError("Invalid parcel number.")

    mask = (data == parcel_num).astype(np.int32)
    mask_img = nib.Nifti1Image(mask, img.affine, img.header)

    return mask, mask_img
