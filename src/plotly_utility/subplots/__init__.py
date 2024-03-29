import itertools
import re

import more_itertools
import numpy as np
import numpy_utility as npu
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots

from ..utils import get_traces_at

__all__ = [
    "hstack",
    "vstack",
    "show_legend_once_for_legend_group",
    "show_empty_subplots",
    "update_xaxes",
    "update_yaxes",
    "vstack_alternately",
    "add_residual_plot",
]


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
    return any(
        np.all(
            np.isin(npu.trunc(annotation_positions, 15), npu.trunc(rough_position, 15)),
            axis=-1,
        )
    )


def _get_sorted_annotations(fig, order=("top to bottom", "left to right")):
    max_len = (
        max(len(a.text) for a in fig.layout.annotations)
        if len(fig.layout.annotations) > 0
        else 1
    )
    annotations = np.array(
        list(
            sorted(
                (
                    (annotation.x, annotation.y, annotation.text)
                    for annotation in fig.layout.annotations
                ),
                key=lambda row: (1 - row[1], row[0]),
            )
        ),
        dtype=[("x", "f8"), ("y", "f8"), ("text", f"U{max_len}")],
    )
    return annotations


def get_subplot_titles(fig, mask_empty_title=True):
    annotations = _get_sorted_annotations(fig)

    titles = np.ma.array(
        [
            (*_get_subplot_title_position(fig, row, col), b"")
            if fig.get_subplot(row, col) is not None
            else (np.nan, np.nan, b"")
            for row, col in fig._get_subplot_coordinates()
        ],
        dtype=annotations.dtype,
    ).reshape(_get_subplot_shape(fig))

    titles_x = npu.trunc(titles["x"], 15)
    titles_y = npu.trunc(titles["y"], 15)
    annotations_x = npu.trunc(annotations["x"], 15)
    annotations_y = npu.trunc(annotations["y"], 15)

    sel = (titles_x[:, :, np.newaxis] == annotations_x[np.newaxis, np.newaxis, :]) & (
        titles_y[:, :, np.newaxis] == annotations_y[np.newaxis, np.newaxis, :]
    )  # (row, col, annotation)

    if mask_empty_title:
        sel &= annotations["text"][np.newaxis, np.newaxis, :] != ""

    row, col, annotation = np.where(sel)

    titles.mask = True
    titles[row, col] = annotations[annotation]
    return titles


def add_subplot_title(fig: go.Figure, text, row, col):
    if _has_subplot_title(fig, row, col):
        raise ValueError(f"fig already has subplot title at row={row} and col={col}")
    title_position = _get_subplot_title_position(fig, row, col)
    fig.add_annotation(
        {
            "y": title_position[1],
            "xref": "paper",
            "x": title_position[0],
            "yref": "paper",
            "text": text,
            "showarrow": False,
            # "font": dict(size=16),
            "xanchor": "center",
            "yanchor": "bottom",
        }
    )


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


def _copy_subplot_ref(
    new_fig, fig, new_row, new_col, row, col, new_fig_max_subplot_ids
):
    new_i_row = new_row - 1
    new_i_col = new_col - 1
    i_row = row - 1
    i_col = col - 1

    if fig._grid_ref[i_row][i_col] is None:
        return

    subplot_ref = fig._grid_ref[i_row][i_col][0]

    new_fig_max_subplot_ids = {
        k: new_fig_max_subplot_ids[k] + id_ - 1
        for k, id_ in zip(
            ["xaxis", "yaxis"],
            map(lambda a: 1 if len(a) == 5 else int(a[5:]), subplot_ref.layout_keys),
        )
    }

    new_subplot_ref = plotly._subplots._init_subplot_xy(
        new_fig.layout,
        False,
        *(fig.layout[kw]["domain"] for kw in subplot_ref.layout_keys),
        # get_max_subplot_ids(new_fig)
        new_fig_max_subplot_ids,
    )

    n_rows, n_cols = _get_subplot_shape(new_fig)
    if new_i_row < n_rows:
        pass
    elif new_i_row == n_rows:
        new_fig._grid_ref.append([None] * n_cols)
    else:
        raise NotImplementedError

    grid_ref_row = new_fig._grid_ref[new_i_row]

    if new_i_col < n_cols:
        pass
    elif new_i_col == n_cols:
        for i in range(n_rows):
            new_fig._grid_ref[i].append(None)
    else:
        raise NotImplementedError

    grid_ref_row[new_i_col] = new_subplot_ref


def _get_subplot_domains_from_grid_ref(fig):
    def f(subplot_refs):
        if subplot_refs is None:
            return np.nan, np.nan
            # return np.inf, np.inf
        assert len(subplot_refs) == 1
        subplot_ref = subplot_refs[0]
        xaxis, yaxis = subplot_ref.layout_keys
        return (np.mean(fig.layout[xaxis].domain), np.mean(fig.layout[yaxis].domain))

    a = npu.ja.apply(f, fig._grid_ref, 2).astype([("x", "f8"), ("y", "f8")])
    return a


def _validate_grid_ref(fig):
    subplot_domains = _get_subplot_domains_from_grid_ref(fig)

    # print("\nbefore:")
    # print(subplot_domains)

    # min_x = np.nanmin(subplot_domains["x"], axis=0)
    # max_x = np.nanmax(subplot_domains["x"], axis=0)
    # min_y = np.nanmin(subplot_domains["y"], axis=1)
    # max_y = np.nanmax(subplot_domains["y"], axis=1)

    # if np.any(np.isnan(subplot_domains["x"])):
    #     # nan_mask_x = np.any(np.isnan(subplot_domains["x"]), axis=0)
    #     # assert np.all(min_x[nan_mask_x] == max_x[nan_mask_x])
    #     nan_mask_x = np.isnan(subplot_domains["x"])
    #     subplot_domains["x"][nan_mask_x] = np.inf
    #
    # if np.any(np.isnan(subplot_domains["y"])):
    #     nan_mask_y = np.any(np.isnan(subplot_domains["y"]), axis=1)
    #     assert np.all(min_y[nan_mask_y] == max_y[nan_mask_y])
    #     subplot_domains["y"][nan_mask_y] = np.arange(10, 10 + np.count_nonzero(nan_mask_y))[:, np.newaxis]

    nan_mask_x = np.isnan(subplot_domains["x"])
    nan_mask_y = np.isnan(subplot_domains["y"])
    assert np.all(nan_mask_x == nan_mask_y)
    nan_mask = nan_mask_x

    # if not npu.is_sorted(subplot_domains["x"], axis=1):
    #     raise RuntimeError(f"grid_ref domain x must be sorted along axis 1: \n{subplot_domains['x']}")
    # if not npu.is_sorted(subplot_domains["y"][::-1], axis=0):
    #     raise RuntimeError(f"grid_ref domain y must be reversely sorted along axis 0: \n{subplot_domains['y']}")
    #
    # print(subplot_domains)
    #
    # subplot_domain_orders = np.rec.fromarrays([
    #     np.argsort(np.argsort(subplot_domains["x"], axis=1), axis=1),
    #     np.argsort(np.argsort(subplot_domains["y"], axis=0), axis=0)
    # ], names=["x", "y"])
    subplot_domains["y"] = 1 - subplot_domains["y"]  # the left-top edge is the origin
    subplot_domain_orders = np.argsort(subplot_domains[~nan_mask], order=["y", "x"])
    # print(subplot_domain_orders)

    order = np.arange(subplot_domains.size).reshape(subplot_domains.shape)
    order[~nan_mask] = order[~nan_mask][subplot_domain_orders]
    # print(order)

    # order = np.argsort(subplot_domains.flatten(), order=["y", "x"]).reshape(subplot_domains.shape)[::-1]
    # order = np.argsort(subplot_domain_orders.flatten(), order=["y", "x"]).reshape(subplot_domains.shape)[::-1]

    if np.all(order.flatten() == np.arange(order.size)):
        pass
    else:
        n_rows, n_cols = _get_subplot_shape(fig)
        fig._grid_ref = [
            [fig._grid_ref[i_row][i_col] for i_row, i_col in idx_rows]
            for idx_rows in np.stack([order // n_cols, order % n_cols], axis=-1)
        ]

    # print("\nafter")
    # print(_get_subplot_domains_from_grid_ref(fig))

    subplot_domains = _get_subplot_domains_from_grid_ref(fig)

    if not npu.is_sorted(subplot_domains["x"], axis=1):
        raise RuntimeError(
            f"grid_ref domain x must be sorted along axis 1: \n{subplot_domains['x']}"
        )
    if not npu.is_sorted(subplot_domains["y"][::-1], axis=0):
        raise RuntimeError(
            f"grid_ref domain y must be reversely sorted along axis 0: \n{subplot_domains['y']}"
        )


def _scale_all_objects(fig, side, fraction=0.5, spacing=None):
    if side in ("top", "bottom"):
        if spacing is None:
            vertical_spacing = default_total_vertical_spacing / 2
        else:
            vertical_spacing = spacing

        # positions old domains
        for annotation in fig.select_annotations(dict(yref="paper")):
            annotation.y = scale_y(annotation.y, fraction, vertical_spacing, side)
        for image in fig.select_layout_images(dict(yref="paper")):
            image.y = scale_y(image.y, fraction, vertical_spacing, side)
            image.sizey = scale_y(image.sizey, fraction, vertical_spacing, "top")
        for trace in fig.data:
            if hasattr(trace.marker, "colorbar"):
                if (
                    trace.marker.colorbar is None
                    or len(trace.marker.colorbar.to_plotly_json()) == 0
                ):
                    continue
                if trace.marker.colorbar.y is None:
                    trace.marker.colorbar.y = 0.5
                if trace.marker.colorbar.len is None:
                    trace.marker.colorbar.len = 1
                trace.marker.colorbar.y = scale_y(
                    trace.marker.colorbar.y, fraction, vertical_spacing, side
                )
                trace.marker.colorbar.len = scale_y(
                    trace.marker.colorbar.len, fraction, vertical_spacing, "top"
                )

        for row, col in fig._get_subplot_coordinates():
            subplot = fig.get_subplot(row, col)
            if subplot is None:
                continue
            subplot.yaxis.domain = tuple(
                scale_y(e, fraction, vertical_spacing, side)
                for e in subplot.yaxis.domain
            )
    elif side in ("left", "right"):
        if spacing is None:
            horizontal_spacing = default_total_horizontal_spacing / 2
        else:
            horizontal_spacing = spacing

        # positions old domains
        for annotation in fig.select_annotations(dict(xref="paper")):
            annotation.x = scale_x(annotation.x, fraction, horizontal_spacing, side)
        for image in fig.select_layout_images(dict(xref="paper")):
            image.x = scale_x(image.x, fraction, horizontal_spacing, side)
            image.sizex = scale_x(
                image.sizex,
                fraction if side in ("right", "bottom") else 1 - fraction,
                horizontal_spacing,
                "right",
            )
        for trace in fig.data:
            if (
                trace.marker.colorbar is None
                or len(trace.marker.colorbar.to_plotly_json()) == 0
            ):
                continue
            if trace.marker.colorbar.x is None:
                trace.marker.colorbar.x = 1
            trace.marker.colorbar.x = scale_x(
                trace.marker.colorbar.x, fraction, horizontal_spacing, side
            )

        for row, col in fig._get_subplot_coordinates():
            subplot = fig.get_subplot(row, col)
            if subplot is None:
                continue
            subplot.xaxis.domain = tuple(
                scale_x(e, fraction, horizontal_spacing, side)
                for e in subplot.xaxis.domain
            )
    else:
        assert side in ("top", "bottom", "right", "left")


def _get_opposite_side(side):
    if side in ("top", "bottom"):
        return "bottom" if side == "top" else "top"
    elif side in ("left", "right"):
        return "right" if side == "left" else "left"
    else:
        raise ValueError("Available side: top, bottom, right or left")


def combine_subplots(new_fig, fig, side, fraction=0.5, spacing=None):
    if side not in ("top", "bottom", "right", "left"):
        raise ValueError("Available side: top, bottom, right or left")

    if side in ("top", "left"):
        new_fig, fig = fig, new_fig

    new_fig = copy(new_fig)
    _scale_all_objects(new_fig, side, fraction, spacing)
    fig = copy(fig)
    _scale_all_objects(fig, _get_opposite_side(side), fraction, spacing)
    n_rows, n_cols = _get_subplot_shape(new_fig)

    if side in ("top", "bottom"):
        subplot_coordinates2 = fig._get_subplot_coordinates()
        subplot_coordinates1 = (
            (n_rows + row, col) for row, col in fig._get_subplot_coordinates()
        )
        if side == "top":
            subplot_coordinates2, subplot_coordinates1 = (
                subplot_coordinates2,
                subplot_coordinates1,
            )
    else:
        subplot_coordinates2 = fig._get_subplot_coordinates()
        subplot_coordinates1 = (
            (row, n_cols + col) for row, col in fig._get_subplot_coordinates()
        )
        if side == "left":
            subplot_coordinates2, subplot_coordinates1 = (
                subplot_coordinates2,
                subplot_coordinates1,
            )

    new_fig_max_subplot_ids = get_max_subplot_ids(new_fig)

    for (row1, col1), (row2, col2) in zip(subplot_coordinates1, subplot_coordinates2):
        # print(f"({row2}, {col2}) -> ({row1}, {col1})")
        _copy_subplot_ref(new_fig, fig, row1, col1, row2, col2, new_fig_max_subplot_ids)
        # print(_get_subplot_domains_from_grid_ref(new_fig))

    _validate_grid_ref(new_fig)
    _validate_grid_ref(fig)

    if side in ("top", "bottom"):
        assert side == "bottom"
        for row, col in fig._get_subplot_coordinates():
            add_old_trace_to_new_fig(
                fig,
                new_fig,
                new_fig_max_subplot_ids,
                row=row,
                col=col,
                new_row=n_rows + row,
                new_col=col,
            )
    else:
        for row, col in fig._get_subplot_coordinates():
            add_old_trace_to_new_fig(
                fig,
                new_fig,
                new_fig_max_subplot_ids,
                row=row,
                col=col,
                new_row=row,
                new_col=n_cols + col,
            )

    return new_fig


def get_max_subplot_ids(fig):
    return {
        "xaxis": max(
            1 if kw == "xaxis" else int(kw[5:])
            for kw in dir(fig.layout)
            if kw.startswith("xaxis")
        ),
        "yaxis": max(
            1 if kw == "yaxis" else int(kw[5:])
            for kw in dir(fig.layout)
            if kw.startswith("yaxis")
        ),
    }


def add_old_trace_to_new_fig(
    old_fig, new_fig, new_fig_max_subplot_ids, row, col, new_row, new_col
):
    traces = get_traces_at(old_fig, row=row, col=col)
    new_fig.add_traces(
        traces, rows=[new_row] * len(traces), cols=[new_col] * len(traces)
    )

    # Axes
    if old_fig._grid_ref[row - 1][col - 1] is None:
        return

    subplot_ref = old_fig._grid_ref[row - 1][col - 1][0]
    new_subplot_ref = new_fig._grid_ref[new_row - 1][new_col - 1][0]

    for axis, new_axis in zip(subplot_ref.layout_keys, new_subplot_ref.layout_keys):
        assert axis[0] == new_axis[0]
        axis_dict = {
            k: v
            for k, v in old_fig.layout[axis].to_plotly_json().items()
            if k != "domain"
        }

        for k in ("anchor", "matches", "scaleanchor", "overlaying"):
            if k in axis_dict.keys():
                axis_type = axis_dict[k][:1]
                axis_id = 1 if len(axis_dict[k]) == 1 else int(axis_dict[k][1:])
                axis_dict[
                    k
                ] = f"{axis_type}{new_fig_max_subplot_ids[f'{axis_type}axis'] + axis_id}"

        new_fig.layout[new_axis].update(axis_dict)

    ref_pattern = re.compile(r"^(x|y)([2-9]|[1-9][0-9]+)?( domain)?$")

    matched = ref_pattern.match(subplot_ref.trace_kwargs["xaxis"])
    target_xaxis_id = int(matched[2] if matched[2] is not None else 1)
    matched = ref_pattern.match(subplot_ref.trace_kwargs["yaxis"])
    target_yaxis_id = int(matched[2] if matched[2] is not None else 1)

    def iter_with_new_refs(objs):
        for obj in objs:
            if obj.xref is None:
                obj.xref = "x"
            if obj.yref is None:
                obj.yref = "y"

            matched_xref = ref_pattern.match(obj.xref)
            if (matched_xref is not None) or ref_pattern.match(obj.yref):
                matched_yref = ref_pattern.match(obj.yref)
                assert matched_yref is not None

                new_obj = copy(obj)
                xaxis_id = int(matched_xref[2] if matched_xref[2] is not None else 1)
                yaxis_id = int(matched_yref[2] if matched_yref[2] is not None else 1)
                if (xaxis_id != target_xaxis_id) or (yaxis_id != target_yaxis_id):
                    continue

                new_obj.xref = f"x{get_max_subplot_ids(new_fig)[f'xaxis'] - get_max_subplot_ids(old_fig)[f'xaxis'] + xaxis_id}"
                if matched_xref[3]:
                    new_obj.xref += matched_xref[3]

                new_obj.yref = f"y{get_max_subplot_ids(new_fig)[f'yaxis'] - get_max_subplot_ids(old_fig)[f'yaxis'] + yaxis_id}"
                if matched_yref[3]:
                    new_obj.yref += matched_yref[3]

                yield new_obj

    for new_image in iter_with_new_refs(old_fig.layout.images):
        new_fig.add_layout_image(new_image)
    for new_annotation in iter_with_new_refs(old_fig.layout.annotations):
        new_fig.add_annotation(new_annotation)
    for new_shape in iter_with_new_refs(old_fig.layout.shapes):
        new_fig.add_shape(new_shape)

    _validate_grid_ref(new_fig)


def get_new_fit_results(new_fig, fit_results, side):
    new_fit_results = np.ma.empty(
        (*_get_subplot_shape(new_fig), *fit_results.shape[2:]), dtype=fit_results.dtype
    )
    new_fit_results.mask = True
    if side == "top":
        nfr = new_fit_results[: fit_results.shape[0]]
    elif side == "bottom":
        nfr = new_fit_results[-fit_results.shape[0] :]
    elif side == "left":
        nfr = new_fit_results[:, : fit_results.shape[1]]
    elif side == "right":
        nfr = new_fit_results[:, -fit_results.shape[1] :]
    else:
        assert False

    if side in ("left", "right"):
        sel = np.arange(nfr.size).reshape(nfr.shape) < np.prod(fit_results.shape)
    else:
        sel = np.arange(nfr.size).reshape(nfr.shape[::-1]).T < np.prod(
            fit_results.shape
        )

    nfr[sel] = fit_results.flatten()
    return new_fit_results


def vstack(fig, *other_fig, fraction=0.5, vertical_spacing=None):
    new_fig = copy(fig)
    figs = other_fig
    # assert len(figs) == 1

    # new_fig_shape = _get_subplot_shape(new_fig)

    if vertical_spacing is None:
        vertical_spacing = default_total_vertical_spacing / 2

    for i, fig in enumerate(figs):
        if i > 0:
            fraction = (i + 1) / (i + 2)
            vertical_spacing = 2 * vertical_spacing / (vertical_spacing + 2) * fraction
            fraction = 1 - fraction

        new_fig = combine_subplots(new_fig, fig, "bottom", fraction, vertical_spacing)
        fig = copy(fig)
        _scale_all_objects(fig, "top", fraction, vertical_spacing)
        # Images
        for image in fig.select_layout_images(selector=dict(yref="paper")):
            new_fig.add_layout_image(copy(image))

        # Annotations
        for annotation in fig.select_annotations(selector=dict(yref="paper")):
            new_fig.add_annotation(copy(annotation))

        # Shapes
        for shape in fig.select_shapes(selector=dict(yref="paper")):
            new_fig.add_shape(copy(shape))

        # Fit Results
        if hasattr(fig, "_fit_results"):
            if not hasattr(new_fig, "_fit_results"):
                new_fig._fit_results = dict()

            for new_fit_name in list(new_fig._fit_results):
                new_fit_results = new_fig._fit_results.pop(new_fit_name)
                new_fig._fit_results[new_fit_name] = get_new_fit_results(
                    new_fig, new_fit_results, "top"
                )

            for fit_name, fit_results in fig._fit_results.items():
                new_fit_results = get_new_fit_results(new_fig, fit_results, "bottom")
                if fit_name in new_fig._fit_results:
                    new_fig._fit_results[fit_name][
                        -fit_results.shape[0] :
                    ] = new_fit_results[-fit_results.shape[0] :]
                else:
                    new_fig._fit_results[fit_name] = new_fit_results

    return new_fig


def hstack(fig, *other_figs, fraction=0.5, horizontal_spacing=None) -> go.Figure:
    new_fig = copy(fig)
    # figs = [other_fig]
    figs = other_figs
    # assert len(figs) == 1

    # new_fig_shape = _get_subplot_shape(new_fig)

    if horizontal_spacing is None:
        horizontal_spacing = default_total_horizontal_spacing / 2

    for i, fig in enumerate(figs):
        # n_rows, n_cols = _get_subplot_shape(fig)
        if i > 0:
            fraction = (i + 1) / (i + 2)
            horizontal_spacing = (
                2 * horizontal_spacing / (horizontal_spacing + 2) * fraction
            )

        new_fig = combine_subplots(new_fig, fig, "right", fraction, horizontal_spacing)

        fig = copy(fig)
        _scale_all_objects(fig, "left", fraction, horizontal_spacing)

        # Images
        for image in fig.select_layout_images(selector=dict(xref="paper")):
            new_image = go.layout.Image(image)
            new_fig.add_layout_image(new_image)

        # Annotations
        for annotation in fig.select_annotations(selector=dict(xref="paper")):
            new_annotation = go.layout.Annotation(annotation)
            new_fig.add_annotation(new_annotation)

        # Shapes
        for shape in fig.select_shapes(selector=dict(xref="paper")):
            new_shape = go.layout.Shape(shape)
            new_fig.add_shape(new_shape)

        # Fit Results
        if hasattr(fig, "_fit_results"):
            if not hasattr(new_fig, "_fit_results"):
                new_fig._fit_results = dict()

            for new_fit_name in list(new_fig._fit_results):
                new_fit_results = new_fig._fit_results.pop(new_fit_name)
                new_fig._fit_results[new_fit_name] = get_new_fit_results(
                    new_fig, new_fit_results, "left"
                )

            for fit_name, fit_results in fig._fit_results.items():
                new_fit_results = get_new_fit_results(new_fig, fit_results, "right")
                if fit_name in new_fig._fit_results:
                    new_fig._fit_results[fit_name][
                        :, -fit_results.shape[0] :
                    ] = new_fit_results[:, -fit_results.shape[0] :]
                else:
                    new_fig._fit_results[fit_name] = new_fit_results

    return new_fig


def get_subplot_coordinates(
    fig,
    x_order="left to right",
    y_order="top to bottom",
    flatten=True,
    mask_empty_subplots=False,
):
    # a = np.array(
    #     list(fig._get_subplot_coordinates()), dtype=[("row", "i8"), ("col", "i8")]
    # ).reshape(*_get_subplot_shape(fig))
    n_rows, n_cols, *_ = np.array(fig._grid_ref, dtype="O").shape
    a = np.array(
        [
            (row, col)
            for row, col, *_ in sorted(
                (
                    (
                        row,
                        col,
                        np.mean(subplot.xaxis.domain),
                        np.mean(subplot.yaxis.domain),
                    )
                    if subplot is not None
                    else (row, col, np.nan, np.nan)
                    for row, col, subplot in (
                        (row, col, fig.get_subplot(row, col))
                        for row, col in fig._get_subplot_coordinates()
                    )
                ),
                key=lambda row: (1 - row[-1], row[-2]),
            )
        ],
        dtype=[("row", "i8"), ("col", "i8")],
    ).reshape(n_rows, n_cols)

    if x_order == "left to right":
        pass
    elif x_order == "right to left":
        a = a[:, ::-1]
    else:
        raise ValueError(x_order)

    if y_order == "top to bottom":
        pass
    elif y_order == "bottom to top":
        a = a[::-1]
    else:
        raise ValueError(y_order)

    if mask_empty_subplots:
        a = a.view(np.ma.MaskedArray)
        a.mask = np.array(
            [
                next(fig.select_traces(row=row, col=col), None) is None
                for row, col in a.flatten()
            ]
        ).reshape(a.shape)

    if flatten:
        return a.flatten()
    else:
        return a


def copy(obj):
    if isinstance(obj, go.Figure):
        import copy

        copied = go.Figure(obj)
        copied._grid_ref = copy.deepcopy(copied._grid_ref)
        if hasattr(obj, "_fit_results"):
            copied._fit_results = copy.deepcopy(obj._fit_results)

        for trace, trace_copied in zip(
            obj.select_traces(selector=dict(type="bar")),
            copied.select_traces(selector=dict(type="bar")),
        ):
            if hasattr(trace, "_x"):
                trace_copied._x = trace._x
        return copied
    elif isinstance(obj, go.layout.Annotation):
        return go.layout.Annotation(obj)
    elif isinstance(obj, go.layout.Shape):
        return go.layout.Shape(obj)
    elif isinstance(obj, go.layout.Image):
        return go.layout.Image(obj)
    else:
        raise TypeError(type(obj))


def show_legend_once_for_legend_group(fig):
    # fig.update_traces(showlegend=False)
    # fig.update_traces(showlegend=True, selector=dict(legendgroup=None))
    for legendgroup in set(filter(None, (trace.legendgroup for trace in fig.data))):
        data = list(
            fig.select_traces(selector=dict(legendgroup=legendgroup, showlegend=True))
        ) + list(
            fig.select_traces(selector=dict(legendgroup=legendgroup, showlegend=None))
        )
        if len(data) == 0:
            data = list(
                fig.select_traces(
                    selector=dict(legendgroup=legendgroup, showlegend=False)
                )
            )
            if len(data) == 0:
                return fig
            else:
                data[0].update(showlegend=True)
                for trace in data[1:]:
                    trace.update(showlegend=False)
        else:
            data[0].update(showlegend=True)
            for trace in data[1:]:
                trace.update(showlegend=False)
        # trace = next(fig.select_traces(dict(legendgroup=legendgroup)))
        # trace.showlegend = True
    return fig


def update_xaxes(fig, target="inside", **kwargs):
    a = get_subplot_coordinates(
        fig,
        x_order="left to right",
        y_order="bottom to top",
        flatten=False,
        mask_empty_subplots=True,
    )
    sel = ~a.mask.view(("?", 2))[..., 0]
    np.put_along_axis(sel, np.argmax(sel, axis=0)[np.newaxis], False, axis=0)
    for row, col in a[sel]:
        fig.update_xaxes(**kwargs, row=row, col=col)
    return fig


def update_yaxes(fig, target="inside", **kwargs):
    a = get_subplot_coordinates(
        fig,
        x_order="left to right",
        y_order="bottom to top",
        flatten=False,
        mask_empty_subplots=True,
    )
    sel = ~a.mask.view(("?", 2))[..., 0]
    np.put_along_axis(sel, np.argmax(sel, axis=1)[:, np.newaxis], False, axis=1)
    for row, col in a[sel]:
        fig.update_yaxes(**kwargs, row=row, col=col)
    return fig


def show_empty_subplots(fig):
    for row, col in fig._get_subplot_coordinates():
        if next(fig.select_traces(row=row, col=col), None) is None:
            if fig.get_subplot(row, col) is not None:
                fig.add_trace(
                    dict(
                        name="_dummy_for_empty_subplots", x=[], y=[], showlegend=False
                    ),
                    row=row,
                    col=col,
                )


def vstack_alternately(
    upper_fig,
    lower_fig,
    small_spacing_fraction=0.1,
    vertical_spacing=0.1,
    horizontal_spacing=0.08,
    subplot_titles=None,
    x_axes_label="x",
):
    """
    Actual small spacing = vertical_spacing * small_spacing_fraction
    """
    upper_fig_shape = _get_subplot_shape(upper_fig)
    lower_fig_shape = _get_subplot_shape(lower_fig)
    if upper_fig_shape[1] != lower_fig_shape[1]:
        raise ValueError("the shapes of upper_fig and lower_fig mismatched.")
    fig_shape = (upper_fig_shape[0] + lower_fig_shape[0], upper_fig_shape[1])

    if subplot_titles is not None:
        subplot_titles = np.ravel(
            list(more_itertools.roundrobin([[""] * 4] * 4, subplot_titles[::-1]))
        )

    fig = plotly.subplots.make_subplots(
        rows=fig_shape[0],
        cols=fig_shape[1],
        row_heights=[0.5] * fig_shape[0],
        column_widths=[1] * fig_shape[1],
        start_cell="bottom-left",
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        subplot_titles=subplot_titles,
    )

    traces_to_be_added = []
    for row, col in upper_fig._get_subplot_coordinates():
        for upper_trace, lower_trace in zip(
            upper_fig.select_traces(row=row, col=col),
            lower_fig.select_traces(row=row, col=col),
        ):
            traces_to_be_added.append((upper_trace, 2 * row, col))
            traces_to_be_added.append((lower_trace, 2 * row - 1, col))

        upper_subplot = plotly._subplots._get_grid_subplot(fig, 2 * row, col)
        lower_subplot = plotly._subplots._get_grid_subplot(fig, 2 * row - 1, col)
        upper_subplot.xaxis.update(
            # matches=lower_subplot.yaxis.anchor,
            matches="x",
            showticklabels=False,
        )
        upper_yaxis = next(upper_fig.select_yaxes(row=row, col=col))
        lower_yaxis = next(lower_fig.select_yaxes(row=row, col=col))
        lowest_row = get_subplot_coordinates(lower_fig)["row"][-1]
        lowest_xaxis = next(lower_fig.select_xaxes(row=lowest_row, col=col))

        upper_subplot.yaxis.update({k: upper_yaxis[k] for k in ("title", "ticksuffix")})
        lower_subplot.yaxis.update({k: lower_yaxis[k] for k in ("title", "ticksuffix")})
        lower_subplot.xaxis.update(
            {k: lowest_xaxis[k] for k in ("title", "ticksuffix")}
        )
        full_height = upper_subplot.yaxis.domain[0] - lower_subplot.yaxis.domain[1]
        upper_domain = list(upper_subplot.yaxis.domain)
        lower_domain = list(lower_subplot.yaxis.domain)
        upper_domain[0] -= (1 - small_spacing_fraction) * full_height / 2
        lower_domain[1] += (1 - small_spacing_fraction) * full_height / 2
        upper_subplot.yaxis.domain = tuple(upper_domain)
        lower_subplot.yaxis.domain = tuple(lower_domain)

    traces_to_be_added = sorted(traces_to_be_added, key=lambda arg: arg[0].name)
    for trace, row, col in traces_to_be_added:
        fig.add_trace(trace, row=row, col=col)

    show_legend_once_for_legend_group(fig)
    return fig


def add_residual_plot(fig, target_name, layout_kwargs=None, log_scale=None):
    if log_scale is None:
        is_log = fig.layout.yaxis.type == "log"
    else:
        assert log_scale in (True, False)
        is_log = log_scale

    traces = {trace.name: trace for trace in fig.data}
    colors = {trace.name: trace.marker.color for trace in fig.data}

    if target_name not in traces.keys():
        raise ValueError(
            f"target_name '{target_name}' is not found in traces of fig: {', '.join(traces.keys())}"
        )

    obs_trace = traces.pop(target_name)
    x = obs_trace.x
    assert all(np.all(x == trace.x) for trace in traces.values())
    obs_y = obs_trace.y
    # error_y = np.sqrt(obs_y)
    error_y = obs_trace.error_y.array
    if error_y is None:
        error_y = itertools.repeat(np.nan)
        is_error_y_valid = False
    else:
        is_error_y_valid = True

    assert np.all(fig.data[0].x == fig.data[1].x)

    if is_log:
        residual_data = npu.to_tidy_data(
            {
                k: zip(x, np.log10(v.y / obs_y), np.log10(np.e) * error_y / obs_y)
                for k, v in traces.items()
            },
            "type",
            ["x", "log(Ratio)", "error_y"],
        )
        residual_fig = px.scatter(
            residual_data,
            x="x",
            y="log(Ratio)",
            error_y="error_y" if is_error_y_valid else None,
            color="type",
            category_orders={"type": list(colors.keys())},
            color_discrete_sequence=list(colors.values()),
        )
    else:
        residual_data = npu.to_tidy_data(
            {k: zip(x, v.y - obs_y, error_y) for k, v in traces.items()},
            "type",
            ["x", "Residual", "error_y"],
        )
        residual_fig = px.scatter(
            residual_data,
            x="x",
            y="Residual",
            error_y="error_y" if is_error_y_valid else None,
            color="type",
            category_orders={"type": list(colors.keys())},
            color_discrete_sequence=list(colors.values()),
        )

    residual_fig.update_xaxes(fig.layout.xaxis).update_yaxes(
        rangemode="tozero"
    ).update_traces(showlegend=False)

    if layout_kwargs is not None:
        residual_fig.update_layout(layout_kwargs)

    fig = vstack(fig, residual_fig, fraction=0.2, vertical_spacing=0.01)
    fig.layout.xaxis.update(title=None, showticklabels=False, matches="x2")
    return fig
