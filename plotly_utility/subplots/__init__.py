import itertools
import warnings

import plotly.subplots
import plotly.graph_objs as go
import numpy as np
import numpy_utility as npu


def _get_subplot_shape(fig: go.Figure):
    if not fig._has_subplots():
        raise ValueError("fig has no subplots")
    n_rows = len(fig._grid_ref)
    n_cols = len(fig._grid_ref[0])
    return n_rows, n_cols


def _get_subplot_title_position(fig, row, col):
    subplot = fig.get_subplot(row, col)
    if subplot is None:
        return np.nan, np.nan
    return np.mean(subplot.xaxis.domain), subplot.yaxis.domain[1]


def _has_subplot_title(fig, row, col):
    rough_position = _get_subplot_title_position(fig, row, col)
    annotations = _get_sorted_annotations(fig)
    annotation_positions = np.c_[annotations["x"], annotations["y"]]
    return any(np.all(np.isin(npu.trunc(annotation_positions, 15), npu.trunc(rough_position, 15)), axis=-1))


def _get_sorted_annotations(fig, order=["top to bottom", "left to right"]):
    max_len = max(len(a.text) for a in fig.layout.annotations) if len(fig.layout.annotations) > 0 else 1
    annotations = np.array(
        list(sorted(
            ((annotation.x, annotation.y, annotation.text) for annotation in fig.layout.annotations),
            key=lambda row: (1 - row[1], row[0])
        )),
        dtype=[("x", "f8"), ("y", "f8"), ("text", f"U{max_len}")]
    )
    return annotations


def get_subplot_titles(fig):
    rows, cols = fig._get_subplot_rows_columns()

    annotations = _get_sorted_annotations(fig)

    titles = np.ma.array([
        [
            (*_get_subplot_title_position(fig, row, col), b"")
            if fig.get_subplot(row, col) is not None else (np.nan, np.nan, b"")
            for col in cols
        ]
        for row in rows
    ], dtype=annotations.dtype)

    titles_x = npu.trunc(titles["x"], 15)
    titles_y = npu.trunc(titles["y"], 15)
    annotations_x = npu.trunc(annotations["x"], 15)
    annotations_y = npu.trunc(annotations["y"], 15)

    sel = np.any(
        (titles_x[..., np.newaxis] == annotations_x[np.newaxis, np.newaxis, :]) &
        (titles_y[..., np.newaxis] == annotations_y[np.newaxis, np.newaxis, :]),
        axis=-1
    )

    titles.mask = True
    titles[sel] = annotations[np.isin(annotations_x, titles_x) & np.isin(annotations_y, titles_y)]
    return titles


def add_subplot_title(fig: go.Figure, text, row, col):
    if _has_subplot_title(fig, row, col):
        raise ValueError(f"fig already has subplot title at row={row} and col={col}")
    title_position = _get_subplot_title_position(fig, row, col)
    fig.add_annotation({
        "y": title_position[1],
        "xref": "paper",
        "x": title_position[0],
        "yref": "paper",
        "text": text,
        "showarrow": False,
        # "font": dict(size=16),
        "xanchor": "center",
        "yanchor": "bottom",
    })


default_total_vertical_spacing = 0.3
default_total_horizontal_spacing = 0.2


def scale_x(x, fraction, horizontal_spacing, spacing_from):
    if spacing_from == "left":
        return 1 - (1 - fraction - horizontal_spacing / 2) * (1 - x)
    elif spacing_from == "right":
        return (fraction - horizontal_spacing / 2) * x
    else:
        raise ValueError(spacing_from)


def scale_y(y, fraction, vertical_spacing, spacing_from):
    if spacing_from == "bottom":
        return 1 - (1 - fraction - vertical_spacing / 2) * (1 - y)
    elif spacing_from == "top":
        return (fraction - vertical_spacing / 2) * y
    else:
        raise ValueError(spacing_from)


def extend_subplot(fig: go.Figure, n_subplots, side="bottom", fraction=0.5,
                   subplot_titles=None,
                   vertical_spacing=None, horizontal_spacing=None):
    n_rows, n_cols = _get_subplot_shape(fig)

    if side in ("top", "bottom"):
        if n_subplots > n_cols:
            raise NotImplementedError(f"{n_subplots} > {n_cols}")

        if vertical_spacing is None:
            vertical_spacing = default_total_vertical_spacing / 2
        if horizontal_spacing is None:
            horizontal_spacing = default_total_horizontal_spacing / n_subplots

        if side == "bottom":
            # positions old domains
            for annotation in fig.layout.annotations:
                if annotation.yref == "paper":
                    annotation.y = scale_y(annotation.y, fraction, vertical_spacing, side)
            for image in fig.layout.images:
                if image.yref == "paper":
                    image.y = scale_y(image.y, fraction, vertical_spacing, side)

            for row, col in fig._get_subplot_coordinates():
                subplot = fig.get_subplot(row, col)
                if subplot is None:
                    continue
                subplot.yaxis.domain = tuple(scale_y(e, fraction, vertical_spacing, side) for e in subplot.yaxis.domain)
        else:
            raise NotImplementedError(side)

        edges = np.linspace(0, 1 - horizontal_spacing * (n_subplots - 1), n_subplots + 1)
        left_edges = edges[:-1] + np.arange(len(edges[:-1])) * horizontal_spacing
        right_edges = edges[1:] + np.arange(len(edges[1:])) * horizontal_spacing
        new_x_domains = zip(left_edges, right_edges)
        new_y_domains = itertools.repeat((0, fraction - vertical_spacing / 2))

        new_grid_ref_row = [
            plotly.subplots._init_subplot_xy(fig.layout, False, x_domain, y_domain, get_max_subplot_ids(fig))
            for x_domain, y_domain in zip(new_x_domains, new_y_domains)
        ] + [None] * (n_cols - n_subplots)
        fig._grid_ref.append(new_grid_ref_row)

        new_rows = itertools.repeat(n_rows + 1)
        new_cols = range(1, n_subplots + 1)
    elif side in ("right", "left"):
        if n_subplots > n_rows:
            raise NotImplementedError(f"{n_subplots} > {n_rows}")

        if vertical_spacing is None:
            vertical_spacing = default_total_vertical_spacing / n_subplots
        if horizontal_spacing is None:
            horizontal_spacing = default_total_horizontal_spacing / 2

        if side == "right":
            # positions old domains
            for annotation in fig.layout.annotations:
                if annotation.yref == "paper":
                    annotation.x = scale_x(annotation.x, fraction, horizontal_spacing, side)
            for image in fig.layout.images:
                if image.yref == "paper":
                    image.x = scale_x(image.x, fraction, horizontal_spacing, side)

            for row, col in fig._get_subplot_coordinates():
                subplot = fig.get_subplot(row, col)
                if subplot is None:
                    continue
                subplot.xaxis.domain = tuple(scale_x(e, fraction, horizontal_spacing, side) for e in subplot.xaxis.domain)
        else:
            raise NotImplementedError(side)

        edges = np.linspace(0, 1 - vertical_spacing * (n_subplots - 1), n_subplots + 1)
        left_edges = edges[:-1] + np.arange(len(edges[:-1])) * vertical_spacing
        right_edges = edges[1:] + np.arange(len(edges[1:])) * vertical_spacing
        new_x_domains = itertools.repeat((fraction + horizontal_spacing / 2, 1))
        new_y_domains = zip(left_edges, right_edges)

        new_grid_ref_col = [
            plotly.subplots._init_subplot_xy(fig.layout, False, x_domain, y_domain, get_max_subplot_ids(fig))
            for x_domain, y_domain in zip(new_x_domains, new_y_domains)
        ] + [None] * (n_rows - n_subplots)
        for i, new_row in enumerate(new_grid_ref_col):
            fig._grid_ref[i].append(new_row)

        new_rows = itertools.repeat(n_rows + 1)
        new_cols = range(1, n_subplots + 1)
    else:
        raise NotImplementedError(side)

    if subplot_titles is not None:
        for row, col, title in zip(new_rows, new_cols, subplot_titles):
            add_subplot_title(fig, title, row, col)

    def _grid_str():
        raise NotImplementedError
    fig._grid_str = _grid_str

    return fig


def get_max_subplot_ids(fig):
    return {
        "xaxis": max(
            1 if kw == "xaxis" else int(kw[5:]) for kw in dir(fig.layout) if kw.startswith("xaxis")
        ),
        "yaxis": max(
            1 if kw == "yaxis" else int(kw[5:]) for kw in dir(fig.layout) if kw.startswith("yaxis")
        )
    }


def add_old_trace_to_new_fig(fig, new_fig, row, col, new_row, new_col):
    from .. import get_traces_at
    new_max_subplot_ids = get_max_subplot_ids(new_fig)

    traces = get_traces_at(fig, row=row, col=col)

    new_fig.add_traces(
        traces,
        rows=[new_row] * len(traces), cols=[new_col] * len(traces)
    )

    # Axes

    subplot_ref = fig._grid_ref[row-1][col-1][0]
    new_subplot_ref = new_fig._grid_ref[new_row - 1][new_col - 1][0]

    new_subplot_axes_ids = {
        k: (1 if len(v) == 1 else int(v[1:]))
        for k, v in new_subplot_ref.trace_kwargs.items()
    }
    assert new_subplot_axes_ids == {
        k: new_max_subplot_ids[k] + (1 if len(v) == 1 else int(v[1:])) - 1
        for k, v in subplot_ref.trace_kwargs.items()
    }

    def old_to_new_axis_id(axis_id: str):
        axis_type = axis_id[:1]
        new_id = (int(axis_id[1:]) if axis_id[1:] != "" else 1) + new_max_subplot_ids[f"{axis_type}axis"] - 1
        if new_id == 1:
            return axis_type
        else:
            return f"{axis_type}{new_id}"

    for axis, new_axis in zip(
        (fig.layout[k] for k in subplot_ref.layout_keys),
        (new_fig.layout[k] for k in new_subplot_ref.layout_keys)
    ):
        axis_dict = {k: v for k, v in axis.to_plotly_json().items() if k not in ("domain",)}
        for k in ("anchor", "matches", "scaleanchor", "overlaying"):
            if k in axis_dict.keys():
                axis_dict[k] = old_to_new_axis_id(axis_dict[k])
        new_axis.update(axis_dict)


def vstack(fig, other_fig, fraction=0.5, vertical_spacing=None):
    new_fig = fig
    figs = [other_fig]
    assert len(figs) == 1

    new_fig_shape = _get_subplot_shape(new_fig)

    if vertical_spacing is None:
        vertical_spacing = default_total_vertical_spacing / 2

    for fig in figs:
        n_rows, _ = _get_subplot_shape(fig)
        subplot_titles = get_subplot_titles(fig)
        assert n_rows == 1

        for i_row in range(n_rows):
            n_cols = sum(1 for cell in fig._grid_ref[i_row] if cell is not None)

            extend_subplot(
                new_fig, n_cols, side="bottom",
                subplot_titles=subplot_titles["text"][i_row].compressed(),
                fraction=fraction, vertical_spacing=vertical_spacing
            )

            for i_col in range(n_cols):
                add_old_trace_to_new_fig(
                    fig, new_fig,
                    row=i_row + 1, col=i_col + 1,
                    new_row=new_fig_shape[0] + i_row + 1, new_col=i_col + 1
                )

        # Images
        for image in fig.layout.images:
            new_image = go.layout.Image(image)
            assert new_image.yref == "paper"
            new_image.y = scale_y(new_image.y, fraction, vertical_spacing, "top")
            new_fig.add_layout_image(new_image)

        # Annotations
        for annotation in fig.layout.annotations:
            new_annotation = go.layout.Annotation(annotation)
            assert new_annotation.yref == "paper"
            new_annotation.y = scale_y(new_annotation.y, fraction, vertical_spacing, "top")
            new_fig.add_annotation(new_annotation)

        # Fit Results
        if hasattr(fig, "_fit_results"):
            if not hasattr(new_fig, "_fit_results"):
                new_fig._fit_results = dict()

            for fit_name, fit_results in fig._fit_results.items():
                if fit_name in new_fig._fit_results.keys():
                    warnings.warn("combining fit results not supported yet", RuntimeWarning)
                else:
                    new_fig._fit_results[fit_name] = fit_results

    return new_fig


def hstack(fig, other_fig, fraction=0.5, horizontal_spacing=None):
    new_fig = fig
    figs = [other_fig]
    assert len(figs) == 1

    new_fig_shape = _get_subplot_shape(new_fig)

    if horizontal_spacing is None:
        horizontal_spacing = default_total_horizontal_spacing / 2

    for fig in figs:
        n_rows, n_cols = _get_subplot_shape(fig)
        subplot_titles = get_subplot_titles(fig)
        assert n_cols == 1

        for i_col in range(n_cols):
            n_rows = sum(1 for i_row in range(n_rows) if fig._grid_ref[i_row][i_col] is not None)

            extend_subplot(
                new_fig, n_rows, side="right",
                subplot_titles=subplot_titles["text"][:, i_col].compressed(),
                fraction=fraction, horizontal_spacing=horizontal_spacing
            )

            for i_row in range(n_rows):
                add_old_trace_to_new_fig(
                    fig, new_fig,
                    row=i_row + 1, col=i_col + 1,
                    new_row=i_row + 1, new_col=new_fig_shape[1] + i_col + 1
                )

        # Images
        for image in fig.layout.images:
            new_image = go.layout.Image(image)
            assert new_image.xref == "paper"
            new_image.x = scale_x(new_image.x, fraction, horizontal_spacing, "left")
            new_fig.add_layout_image(new_image)

        # Annotations
        for annotation in fig.layout.annotations:
            new_annotation = go.layout.Annotation(annotation)
            assert new_annotation.xref == "paper"
            new_annotation.x = scale_x(new_annotation.x, fraction, horizontal_spacing, "left")
            new_fig.add_annotation(new_annotation)

        # Fit Results
        if hasattr(fig, "_fit_results"):
            if not hasattr(new_fig, "_fit_results"):
                new_fig._fit_results = dict()

            for fit_name, fit_results in fig._fit_results.items():
                if fit_name in new_fig._fit_results.keys():
                    warnings.warn("combining fit results not supported yet", RuntimeWarning)
                else:
                    new_fig._fit_results[fit_name] = fit_results

    return new_fig



