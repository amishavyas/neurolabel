from importlib.resources import as_file, files
import numpy as np
import nibabel as nib
from nilearn.image import load_img


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
<<<<<<< Updated upstream
=======
        
def atlas_overlap(nifti_file:str, parcel_num:int, atlas_string:str, top_10:bool=True):
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
    atlas = datasets.fetch_atlas_schaefer_2018()
    parcellation_file = nifti_file
    _, mask_img = extract_parcel_mask(parcellation_file, 3)

    atlas_img = image.resample_to_img(
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
    overlap_df2 = overlap_df[overlap_df["intersection_voxels"] > 0] 
    if top_10: 
        overlap_df2 = overlap_df2.iloc[0:10]
    return overlap_df2
>>>>>>> Stashed changes
