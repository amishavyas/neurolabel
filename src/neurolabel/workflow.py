"""High-level orchestration for the neurolabel package."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from .atlas import atlas_overlap, plot_anat
from .functional_decoder import decode_function, plot_decoder
from .utils import summarize_statistical_map

from numbers import Integral

def run_workflow(
    *,
    parcellation: Any,
    statistical_map: Any = None,
    parcel_ids: Optional[
        Union[int, Iterable[int]]
    ] = None,
    threshold: Optional[float] = None,
    mode: str = "two-sided",
    min_voxels: int = 1,
    label_atlas: Any = "schaefer",
    run_anatomical_overlap: bool = True,
    run_functional_decoding: bool = True,
    top_anatomical: Optional[int] = 10,
    top_functional: Optional[int] = 10,
    decoder_kwargs: Optional[
        Dict[str, Any]
    ] = None,
    continue_on_error: bool = True,
) -> Dict[str, object]:
    """Run the complete neurolabel workflow.

    Exactly one of ``statistical_map`` or ``parcel_ids``
    must be supplied.

    The workflow performs the following steps:

    1. Identify parcels from a statistical map, or use the
       supplied parcel IDs.
    2. Run anatomical atlas overlap for each parcel.
    3. Plot the anatomical-overlap results using ``plot_anat``.
    4. Run functional decoding for each parcel.
    5. Plot the decoding results using ``plot_decoder``.
    6. Return all generated DataFrames.

    Parameters
    ----------
    parcellation
        Parcellation passed unchanged to
        ``summarize_statistical_map``, ``atlas_overlap``,
        and ``decode_function``.

    statistical_map
        Statistical image used to identify affected parcels.
        Cannot be supplied together with ``parcel_ids``.

    parcel_ids
        One parcel ID or an iterable of parcel IDs. Cannot
        be supplied together with ``statistical_map``.

    threshold
        Threshold passed directly to
        ``summarize_statistical_map``.

    mode
        Thresholding mode passed directly to
        ``summarize_statistical_map``.

    min_voxels
        Minimum number of significant voxels required for a
        parcel to be included.

    label_atlas
        Atlas passed directly to ``atlas_overlap``.

    run_anatomical_overlap
        Whether to run ``atlas_overlap`` and ``plot_anat``.

    run_functional_decoding
        Whether to run ``decode_function`` and
        ``plot_decoder``.

    top_anatomical
        Number of anatomical regions shown by ``plot_anat``.
        Use ``None`` to plot all returned regions.

    top_functional
        Number of functional terms shown by ``plot_decoder``.
        Use ``None`` to plot all returned terms.

    decoder_kwargs
        Additional keyword arguments passed to
        ``decode_function``.

    continue_on_error
        When True, record an error and continue processing
        the remaining parcels. When False, immediately raise
        the error.

    Returns
    -------
    dict
        Dictionary containing:

        ``input_type``
            Whether the workflow started from a statistical
            map or parcel IDs.

        ``parcel_ids``
            Parcel IDs processed by the workflow.

        ``parcel_summary``
            Statistical-map summary, or a simple parcel-ID
            DataFrame for direct parcel input.

        ``atlas_overlap``
            Dictionary mapping each parcel ID to its atlas
            overlap DataFrame.

        ``functional_decoding``
            Dictionary mapping each parcel ID to its
            functional-decoding DataFrame.

        ``errors``
            DataFrame containing any workflow errors.
    """

    has_statistical_map = statistical_map is not None
    has_parcel_ids = parcel_ids is not None

    if has_statistical_map == has_parcel_ids:
        raise ValueError(
            "Provide exactly one input source: "
            "statistical_map or parcel_ids."
        )

    if (
        not run_anatomical_overlap
        and not run_functional_decoding
    ):
        raise ValueError(
            "At least one workflow stage must be enabled."
        )

    for name, value in {
        "top_anatomical": top_anatomical,
        "top_functional": top_functional,
    }.items():
        if value is not None and value < 1:
            raise ValueError(
                f"{name} must be at least 1 or None."
            )

    decoder_options = dict(decoder_kwargs or {})

    reserved_arguments = {
        "parcel_num",
        "parcellation_file",
    }.intersection(decoder_options)

    if reserved_arguments:
        names = ", ".join(
            sorted(reserved_arguments)
        )
        raise ValueError(
            "decoder_kwargs cannot override "
            "workflow-controlled arguments: "
            f"{names}."
        )

    # --------------------------------------------------
    # Determine which parcel IDs will be processed.
    # --------------------------------------------------

    if has_statistical_map:
        parcel_summary = summarize_statistical_map(
            statistical_map=statistical_map,
            parcellation=parcellation,
            threshold=threshold,
            mode=mode,
            min_voxels=min_voxels,
        )

        if "parcel_id" not in parcel_summary.columns:
            raise ValueError(
                "summarize_statistical_map must return "
                "a 'parcel_id' column."
            )

        selected_parcels = (
            parcel_summary["parcel_id"]
            .dropna()
            .astype(int)
            .tolist()
        )

    else:
        if isinstance(parcel_ids, Integral):
            selected_parcels = [parcel_ids]
        else:
            selected_parcels = list(parcel_ids)

    parcel_summary = pd.DataFrame({"parcel_id": selected_parcels})

    atlas_results: Dict[int, pd.DataFrame] = {}
    functional_results: Dict[int, pd.DataFrame] = {}
    anatomical_figures: Dict[int, object] = {}
    decoder_figures: Dict[int, object] = {}
    errors: List[Dict[str, object]] = []

    # --------------------------------------------------
    # Run the workflow for each selected parcel.
    # --------------------------------------------------

    for parcel_id in selected_parcels:

        # ----------------------------------------------
        # Anatomical atlas overlap
        # ----------------------------------------------

        if run_anatomical_overlap:
            try:
                overlap = atlas_overlap(
                    target_atlas=parcellation,
                    parcel_num=parcel_id,
                    label_atlas=label_atlas,
                )

                atlas_results[parcel_id] = overlap

                if not overlap.empty:
                    anatomical_n = (
                        len(overlap)
                        if top_anatomical is None
                        else min(
                            top_anatomical,
                            len(overlap),
                        )
                    )

                    fig = plot_anat(
                        overlap,
                        top_n=anatomical_n,
                    )
                    
                    anatomical_figures[parcel_id] = fig

            except Exception as exc:
                errors.append(
                    {
                        "parcel_id": parcel_id,
                        "stage": "anatomical_overlap",
                        "error": str(exc),
                    }
                )

                if not continue_on_error:
                    raise

        # ----------------------------------------------
        # Functional decoding
        # ----------------------------------------------

        if run_functional_decoding:
            try:
                decoded = decode_function(
                    parcel_num=parcel_id,
                    parcellation_file=parcellation,
                    **decoder_options,
                )

                functional_results[parcel_id] = decoded

                if not decoded.empty:
                    functional_n = (
                        len(decoded)
                        if top_functional is None
                        else min(
                            top_functional,
                            len(decoded),
                        )
                    )

                    fig = plot_decoder(
                    decoded,
                    top_n=functional_n,
                    )
                    
                    decoder_figures[parcel_id] = fig
                
            except Exception as exc:
                errors.append(
                    {
                        "parcel_id": parcel_id,
                        "stage": "functional_decoding",
                        "error": str(exc),
                    }
                )

                if not continue_on_error:
                    raise

    workflow_rows = []

    for parcel_id in selected_parcels:
        parcel_errors = pd.DataFrame(
            [
                error
                for error in errors
                if error["parcel_id"] == parcel_id
            ],
            columns=[
                "parcel_id",
                "stage",
                "error",
            ],
        )
    
        workflow_rows.append(
            {
                "input_type": (
                    "statistical_map"
                    if has_statistical_map
                    else "parcel_ids"
                ),
                "parcel_id": parcel_id,
                "parcel_summary": parcel_summary.loc[
                    parcel_summary["parcel_id"].eq(parcel_id)
                ].copy(),
                "atlas_overlap": atlas_results.get(
                    parcel_id,
                    pd.DataFrame(),
                ),
                "anatomical_figure": anatomical_figures.get(
                    parcel_id
                ),
                "functional_decoding": functional_results.get(
                    parcel_id,
                    pd.DataFrame(),
                ),
                "decoder_figure": decoder_figures.get(
                    parcel_id
                ),
                "errors": parcel_errors,
            }
        )
    
    return pd.DataFrame(workflow_rows)