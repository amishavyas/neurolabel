import logging
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import nilearn
from nilearn import image, plotting
from .atlas import (
    atlas_overlap,
    plot_anat,
)
from .utils import extract_parcel_mask
from .functional_decoder import (
    decode_function,
    plot_decoder,
)


def interactive_parcellation_viewer(
    parcellation,
    label_atlas="harvard-oxford",
    plot_anat_kwargs={},
    decoder_kwargs={},
    top_n=10,
    dpi=120,
):
    """
    Create an interactive parcel viewer in a Jupyter notebook.

    Parameters
    ----------
    parcellation : Niimg-like object
        Label parcellation containing integer parcel values.
    label_atlas : str, default="harvard-oxford"
        Anatomical atlas passed to `atlas_overlap`. It can be the same as parcellation if it is already labelled. 
    plot_anat_kwargs : dict, optional
        Additional arguments passed to `plot_anat`.
    decoder_kwargs : dict, optional
        Additional arguments passed to `decode_function`.
    top_n : int, default=10
        Number of anatomical regions and decoder terms displayed.
    dpi : int, default=120
        Resolution used when converting static figures to PNG images.

    Notes
    -----
    By default the dashboard shows one row: static ROI, functional decoder,
    and anatomical overlap side by side. Clicking "Open interactive ROI"
    switches to two rows instead: static ROI next to the interactive surface
    viewer on top, and functional decoder next to anatomical overlap below.

    Returns
    -------
    ipywidgets.VBox
        Interactive parcel viewer.
    """
    logging.getLogger("nimare").setLevel(logging.WARNING)

    if isinstance(parcellation, nilearn.datasets.atlas.Atlas):
        atlas_img = image.load_img(parcellation.maps)
    else:
        atlas_img = image.load_img(parcellation)

    atlas_data = atlas_img.get_fdata()

    # Get the set of valid finite, nonzero parcel labels.
    finite_values = atlas_data[np.isfinite(atlas_data)]
    valid_parcels = np.unique(np.round(finite_values).astype(int))
    valid_parcels = valid_parcels[valid_parcels != 0]

    if valid_parcels.size == 0:
        raise ValueError(
            "The atlas does not contain any finite, nonzero parcel labels."
        )

    valid_parcel_set = set(valid_parcels.tolist())
    decoder_cache = {}
    anatomy_cache = {}

    def fig_to_png(fig):
        buf = BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
        buf.seek(0)
        png_data = buf.read()
        buf.close()
        return png_data

    def nilearn_display_to_png(nilearn_display):
        fig = nilearn_display.frame_axes.figure
        png_data = fig_to_png(fig)
        nilearn_display.close()
        plt.close(fig)
        return png_data

    parcel_input = widgets.BoundedIntText(
        description="Parcel number:",
        value=int(valid_parcels[0]),
        min=int(valid_parcels.min()),
        max=int(valid_parcels.max()),
        step=1,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="300px"),
    )

    show_button = widgets.Button(
        description="Show parcel",
        button_style="primary",
    )

    reset_button = widgets.Button(
        description="Show all parcels",
    )

    validation_status = widgets.HTML()

    roi_heading = widgets.HTML("<h3>Parcel ROI</h3>")
    decoder_heading = widgets.HTML("<h3>Functional Decoder</h3>")
    anatomy_heading = widgets.HTML("<h3>Anatomical Overlap</h3>")
    interactive_heading = widgets.HTML("<h3>Interactive ROI</h3>")

    # Static ROI image shown in the three-column dashboard.
    roi_output = widgets.Output(
        layout=widgets.Layout(width="100%", height="auto")
    )

    interactive_button = widgets.Button(
        description="Open interactive ROI",
        icon="cube",
        tooltip="Open the currently displayed ROI on an interactive surface",
    )
    close_interactive_button = widgets.Button(
        description="Close interactive ROI",
        icon="times",
        layout=widgets.Layout(display="none"),
    )
    interactive_status = widgets.HTML()

    # Left at its natural size (not stretched to 100%) since the surface
    # report has its own fixed internal layout (brain view + colorbar +
    # dropdowns); stretching its container just left a lot of empty space.
    interactive_output = widgets.Output(layout=widgets.Layout(display="none"))

    # Tracks whichever atlas/parcel is currently shown by the static plot.
    current_roi = {
        "img": atlas_img,
        "title": "All parcels",
        "all_parcels": True,
    }

    decoder_image = widgets.Image(
        format="png",
        layout=widgets.Layout(
            width="100%",
            height="auto",
            display="none",
        ),
    )

    anatomy_image = widgets.Image(
        format="png",
        layout=widgets.Layout(
            width="100%",
            height="auto",
            display="none",
        ),
    )

    roi_status = widgets.HTML()
    decoder_status = widgets.HTML()
    anatomy_status = widgets.HTML()

    def set_image(image_widget, png_data=None):
        """Show a PNG image, or hide the widget if no image is supplied."""
        if png_data:
            image_widget.value = png_data
            image_widget.layout.display = ""
        else:
            image_widget.value = b""
            image_widget.layout.display = "none"

    def clear_roi_output():
        """Remove the current static or interactive ROI display."""
        roi_output.clear_output(wait=True)

    def display_roi(roi_img, title=None, all_parcels=False):
        """Display the static ROI plot and remember it for the button."""
        current_roi.update(
            img=roi_img,
            title=title,
            all_parcels=all_parcels,
        )
        clear_roi_output()

        with roi_output:
            if all_parcels:
                roi_display = plotting.plot_roi(
                    roi_img=roi_img,
                    title=title,
                    display_mode="ortho",
                    colorbar=False,
                )
            else:
                cut_coords = plotting.find_xyz_cut_coords(roi_img)
                roi_display = plotting.plot_roi(
                    roi_img=roi_img,
                    title=title,
                    display_mode="ortho",
                    cut_coords=cut_coords,
                    threshold=0.5,
                    colorbar=False,
                )

            display(
                widgets.Image(
                    value=nilearn_display_to_png(roi_display),
                    format="png",
                    layout=widgets.Layout(width="100%", height="auto"),
                )
            )

    def open_interactive_roi(_=None):
        """Open the currently displayed ROI in a compact surface viewer."""
        interactive_button.disabled = True
        interactive_status.value = "Loading interactive surface viewer..."
        interactive_output.layout.display = ""
        close_interactive_button.layout.display = ""
        interactive_output.clear_output(wait=True)

        try:
            # Suppress Nilearn dataset/cache messages in the dashboard.
            captured_output = StringIO()
            with redirect_stdout(captured_output), redirect_stderr(captured_output):
                surface_view = plotting.view_img_on_surf(
                    stat_map_img=current_roi["img"],
                    surf_mesh="fsaverage",
                    threshold=(
                        None if current_roi["all_parcels"] else 0.5
                    ),
                    cmap=(
                        "cold_hot"
                        if current_roi["all_parcels"]
                        else "Reds"
                    ),
                    symmetric_cmap=False,
                    title=current_roi["title"],
                )

            # Fixed size that comfortably fits the brain view, colorbar, and
            # hemisphere/view dropdowns without extra padding around them.
            surface_view.resize(480, 380)
            with interactive_output:
                display(surface_view)
            interactive_status.value = ""

            # Switch from a 3-column single row ("ROI - functional -
            # anatomy") to a 2-column grid: "ROI - interactive map" on top,
            # "functional - anatomy" below. Using a real CSS grid (not two
            # independent flex rows) means the column widths are shared
            # between both rows, so the panels actually line up.
            plots_grid.layout.grid_template_columns = (
                "repeat(2, minmax(360px, max-content))"
            )
            plots_grid.children = (
                roi_panel,
                interactive_panel,
                decoder_panel,
                anatomy_panel,
            )
        except Exception as exc:
            interactive_output.clear_output(wait=True)
            interactive_status.value = error_html(
                "Could not create the interactive ROI viewer", exc
            )
        finally:
            interactive_button.disabled = False

    def close_interactive_roi(_=None):
        interactive_output.clear_output(wait=True)
        interactive_output.layout.display = "none"
        close_interactive_button.layout.display = "none"
        interactive_status.value = ""

        # Back to the single 3-column row: "ROI - functional - anatomy".
        plots_grid.layout.grid_template_columns = (
            "repeat(3, minmax(300px, max-content))"
        )
        plots_grid.children = (roi_panel, decoder_panel, anatomy_panel)

    def set_busy(is_busy):
        show_button.disabled = is_busy
        reset_button.disabled = is_busy
        parcel_input.disabled = is_busy
        interactive_button.disabled = is_busy

    def clear_images():
        clear_roi_output()
        close_interactive_roi()
        set_image(decoder_image)
        set_image(anatomy_image)

    def reset_headings():
        roi_heading.value = "<h3>Parcel ROI</h3>"
        decoder_heading.value = "<h3>Functional Decoder</h3>"
        anatomy_heading.value = "<h3>Anatomical Overlap</h3>"

    def error_html(message, exc):
        return (
            f"<span style='color:red'>"
            f"{message}: {type(exc).__name__}: {exc}"
            f"</span>"
        )

    def get_selected_parcel():
        parcel_num = int(parcel_input.value)

        if parcel_num not in valid_parcel_set:
            raise ValueError(
                f"Parcel {parcel_num} is not a valid atlas label. "
                f"Available labels range from "
                f"{valid_parcels.min()} to {valid_parcels.max()}."
            )

        return parcel_num

    def show_all_parcels(_=None):
        set_busy(True)
        validation_status.value = ""
        reset_headings()
        clear_images()

        roi_status.value = "Loading atlas..."
        decoder_status.value = (
            "Select a parcel to display decoder results."
        )
        anatomy_status.value = (
            "Select a parcel to display anatomical overlap."
        )

        try:
            display_roi(
                roi_img=atlas_img,
                title="All parcels",
                all_parcels=True,
            )
            roi_status.value = ""

        except Exception as exc:
            clear_roi_output()
            roi_status.value = error_html(
                "Could not display the atlas",
                exc,
            )

        finally:
            set_busy(False)

    def show_selected_parcel(_=None):
        validation_status.value = ""

        try:
            parcel_num = get_selected_parcel()

        except ValueError as exc:
            validation_status.value = (
                f"<span style='color:red'><b>{exc}</b></span>"
            )
            return

        set_busy(True)
        reset_headings()
        clear_images()

        roi_status.value = "Loading parcel..."
        decoder_status.value = ""
        anatomy_status.value = ""

        try:
            _, parcel_img = extract_parcel_mask(
                atlas_img,
                parcel_num,
            )

            parcel_img = image.load_img(parcel_img)
            parcel_data = parcel_img.get_fdata()

            n_voxels = np.count_nonzero(parcel_data)

            if n_voxels == 0:
                raise ValueError(
                    f"Parcel {parcel_num} contains no nonzero voxels."
                )

            roi_heading.value = (
                f"<h3>Parcel {parcel_num} ROI: "
                f"{n_voxels:,} voxels</h3>"
            )

            try:
                display_roi(
                    roi_img=parcel_img,
                    title=None,
                    all_parcels=False,
                )
                roi_status.value = ""

            except Exception as exc:
                clear_roi_output()
                roi_status.value = error_html(
                    "Could not display the selected parcel",
                    exc,
                )

            anatomy_status.value = (
                "Calculating anatomical overlap..."
            )

            try:
                if parcel_num not in anatomy_cache:
                    anatomy_cache[parcel_num] = atlas_overlap(
                        label_atlas=label_atlas,
                        target_atlas=atlas_img,
                        parcel_num=parcel_num,
                    )

                overlap_df = anatomy_cache[parcel_num]

                if overlap_df is None or overlap_df.empty:
                    set_image(anatomy_image)
                    anatomy_status.value = (
                        "No anatomical overlap was found."
                    )

                else:
                    anatomy_fig = plot_anat(
                        overlap_df,
                        top_n=top_n,
                        **plot_anat_kwargs,
                    )

                    if anatomy_fig is None:
                        set_image(anatomy_image)
                        anatomy_status.value = (
                            "plot_anat returned no figure."
                        )

                    else:
                        set_image(
                            anatomy_image,
                            fig_to_png(anatomy_fig),
                        )
                        anatomy_status.value = ""
                        plt.close(anatomy_fig)

            except Exception as exc:
                set_image(anatomy_image)
                anatomy_status.value = error_html(
                    "Could not create anatomy plot",
                    exc,
                )

            decoder_status.value = (
                "Decoding functional associations. "
                "This may take a while..."
            )

            try:
                if parcel_num not in decoder_cache:
                    captured_output = StringIO()

                    with (
                        redirect_stdout(captured_output),
                        redirect_stderr(captured_output),
                    ):
                        decoder_cache[parcel_num] = decode_function(
                            parcellation_file=atlas_img,
                            parcel_num=parcel_num,
                            **decoder_kwargs,
                        )

                decoded_df = decoder_cache[parcel_num]

                if decoded_df is None or decoded_df.empty:
                    set_image(decoder_image)
                    decoder_status.value = (
                        "No decoder results were found."
                    )

                else:
                    decoder_fig = plot_decoder(
                        decoded_df,
                        top_n=top_n,
                    )

                    if decoder_fig is None:
                        set_image(decoder_image)
                        decoder_status.value = (
                            "plot_decoder returned no figure."
                        )

                    else:
                        set_image(
                            decoder_image,
                            fig_to_png(decoder_fig),
                        )
                        decoder_status.value = ""
                        plt.close(decoder_fig)

            except Exception as exc:
                set_image(decoder_image)
                decoder_status.value = error_html(
                    "Could not create decoder plot",
                    exc,
                )

        except Exception as exc:
            clear_roi_output()
            roi_heading.value = "<h4>Parcel ROI</h4>"
            roi_status.value = error_html(
                "Could not extract the selected parcel",
                exc,
            )

        finally:
            set_busy(False)

    show_button.on_click(show_selected_parcel)
    reset_button.on_click(show_all_parcels)
    interactive_button.on_click(open_interactive_roi)
    close_interactive_button.on_click(close_interactive_roi)

    controls = widgets.HBox(
        [
            parcel_input,
            show_button,
            reset_button,
            interactive_button,
            close_interactive_button,
        ],
        layout=widgets.Layout(
            align_items="center",
            gap="8px",
            flex_flow="row wrap",
        ),
    )

    # Column widths come from the grid itself (see plots_grid below), so
    # panels just need a sensible minimum width, not their own flex sizing.
    roi_panel = widgets.VBox(
        [roi_heading, roi_status, roi_output],
        layout=widgets.Layout(min_width="300px", align_items="center"),
    )
    interactive_panel = widgets.VBox(
        [interactive_heading, interactive_status, interactive_output],
        layout=widgets.Layout(min_width="360px", align_items="center"),
    )
    decoder_panel = widgets.VBox(
        [decoder_heading, decoder_status, decoder_image],
        layout=widgets.Layout(min_width="300px", align_items="center"),
    )
    anatomy_panel = widgets.VBox(
        [anatomy_heading, anatomy_status, anatomy_image],
        layout=widgets.Layout(min_width="300px", align_items="center"),
    )

    # A real CSS grid instead of independent flex rows: column widths are
    # shared across rows, so "ROI over functional" and "interactive over
    # anatomy" actually line up instead of drifting independently. Starts
    # as a single row of 3 columns; open_interactive_roi/close_interactive_roi
    # switch it to a 2-column, 2-row grid and back.
    plots_grid = widgets.GridBox(
        [roi_panel, decoder_panel, anatomy_panel],
        layout=widgets.Layout(
            width="100%",
            grid_template_columns="repeat(3, minmax(300px, max-content))",
            grid_gap="16px",
        ),
    )

    viewer = widgets.VBox(
        [
            controls,
            validation_status,
            plots_grid,
        ],
        layout=widgets.Layout(width="100%"),
    )

    display(viewer)
    show_all_parcels()

    return viewer
