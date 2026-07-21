import numpy as np
import nibabel as nib


def extract_parcel_mask(nifti_file, parcel_num):
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
    img = nib.load(nifti_file)
    data = np.round(img.get_fdata()).astype(int)

    if parcel_num == 0 or parcel_num not in np.unique(data):
        raise ValueError(
            "Invalid parcel number. Please enter a valid integer.")

    mask = (data == parcel_num).astype(np.int32)
    mask_img = nib.Nifti1Image(mask, img.affine, img.header)

    return mask, mask_img
