from neurolabel.utils import extract_parcel_mask
from nimare.extract import fetch_neurosynth
from nimare.decode import discrete
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from functools import lru_cache

plt.rcParams["font.family"] = "Helvetica"

@lru_cache(maxsize=None)
def get_studyset(version, source, vocab, feature_type, target):
    return fetch_neurosynth(
        version=version,
        source=source,
        vocab=vocab,
        type=feature_type,
        return_type="studyset",
        target=target,
    )[0]


def decode_function(
    parcel_num: int,
    *,
    parcellation_file: str | Path,
    version: str = "7",
    source: str = "abstract",
    vocab: str = "terms",
    feature_type: str = "tfidf",
    target: str = "mni152_2mm",
) -> pd.DataFrame:
    """
    Decode a parcellation region using a NiMARE ROI association decoder.

    Parameters
    ----------
    parcel_num
        Integer label identifying the region to decode.
    parcellation_file
        Path to a labeled parcellation NIfTI image.
    version
        Neurosynth dataset version.
    source
        Annotation source.
    vocab
        Annotation vocabulary.
    feature_type
        Annotation feature type.
    target
        Target coordinate space.

    Returns
    -------
    pandas.DataFrame
        Decoding results for the selected region.

    Raises
    ------
    FileNotFoundError
        If ``parcellation_file`` does not exist.
    ValueError
        If ``parcel_num`` is not present in the parcellation.
    """
    _, mask_img = extract_parcel_mask(parcellation_file, parcel_num)

    studyset = get_studyset(
        version=version,
        source=source,
        vocab=vocab,
        feature_type=feature_type,
        target=target,
    )

    feature_columns = [
        column
        for column in studyset.annotations_df.columns
        if "__" in column
    ]

    feature_groups = sorted({
        column.split("__", 1)[0]
        for column in feature_columns
    })

    decoder = discrete.ROIAssociationDecoder(
        mask_img,
        feature_group=feature_groups[0],
    )

    decoder.fit(studyset)

    results = decoder.transform()

    return results


def plot_decoder(decoded_df, top_n=10, bar_color="#F4A300"):
    """
    Plot the top decoding results as a horizontal bar chart.

    Parameters
    ----------
    decoded_df : pandas.DataFrame
        DataFrame returned by ``decode_function`` containing decoding
        statistics.
    top_n : int, default=10
        Number of highest-ranked features to display.
    bar_color : str, default="#F4A300"
        Color of the bars.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the horizontal bar chart.
    """
    top = decoded_df.sort_values("r", ascending=False).head(top_n).copy()

    top.index = top.index.str.replace(
        r"^(?:LDA\d+_abstract_weight__\d+_|terms_abstract_tfidf__)",
        "",
        regex=True,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(top.index, top["r"], color=bar_color)
    ax.invert_yaxis()

    ax.set_xlabel("Correlation (r)", fontsize=13)
    ax.set_ylabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
