import itertools
import re
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


def get_subplot_titles(fig, mask_empty_title=True):
    annotations = _get_sorted_annotations(fig)

    titles = np.ma.array([
        (*_get_subplot_title_position(fig, row, col), b"")
        if fig.get_subplot(row, col) is not None else (np.nan, np.nan, b"")
        for row, col in get_subplot_coordinates(fig).tolist()
    ], dtype=annotations.dtype).reshape(_get_subplot_shape(fig))

    titles_x = npu.trunc(titles["x"], 15)
    titles_y = npu.trunc(titles["y"], 15)
    annotations_x = npu.trunc(annotations["x"], 15)
    annotations_y = npu.trunc(annotations["y"], 15)

    sel = (
        (titles_x[..., np.newaxis] == annotations_x[np.newaxis, np.newaxis, :]) &
        (titles_y[..., np.newaxis] == annotations_y[np.newaxis, np.newaxis, :])
    )  # (row, col, annotation)

    if mask_empty_title:
        sel &= annotations["text"][np.newaxis, np.newaxis, :] != ""

    titles.mask = True
    titles[sel.any(axis=-1)] = annotations[sel.any(axis=1).any(axis=0)]
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


def _copy_subplot_ref(new_fig, fig, new_row, new_col, row, col, new_fig_max_subplot_ids):
    new_i_row = new_row - 1
    new_i_col = new_col - 1
    i_row = row - 1
    i_col = col - 1

    if fig._grid_ref[i_row][i_col] is None:
        return

    subplot_ref = fig._grid_ref[i_row][i_col][0]

    new_fig_max_subplot_ids = {
        k: new_fig_max_subplot_ids[k] + id_ - 1
        for k, id_ in zip(["xaxis", "yaxis"], map(lambda a: 1 if len(a) == 5 else int(a[5:]), subplot_ref.layout_keys))
    }

    new_subplot_ref = plotly.subplots._init_subplot_xy(
        new_fig.layout, False, *(fig.layout[kw]["domain"] for kw in subplot_ref.layout_keys),
        # get_max_subplot_ids(new_fig)
        new_fig_max_subplot_ids
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
                if trace.marker.colorbar is None or len(trace.marker.colorbar.to_plotly_json()) == 0:
                    continue
                if trace.marker.colorbar.y is None:
                    trace.marker.colorbar.y = 0.5
                if trace.marker.colorbar.len is None:
                    trace.marker.colorbar.len = 1
                trace.marker.colorbar.y = scale_y(trace.marker.colorbar.y, fraction, vertical_spacing, side)
                trace.marker.colorbar.len = scale_y(trace.marker.colorbar.len, fraction, vertical_spacing, "top")

        for row, col in fig._get_subplot_coordinates():
            subplot = fig.get_subplot(row, col)
            if subplot is None:
                continue
            subplot.yaxis.domain = tuple(scale_y(e, fraction, vertical_spacing, side) for e in subplot.yaxis.domain)
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
            image.sizex = scale_x(image.sizex, fraction if side in ("right", "bottom") else 1 - fraction, horizontal_spacing, "right")
        for trace in fig.data:
            if trace.marker.colorbar is None or len(trace.marker.colorbar.to_plotly_json()) == 0:
                continue
            if trace.marker.colorbar.x is None:
                trace.marker.colorbar.x = 1
            trace.marker.colorbar.x = scale_x(trace.marker.colorbar.x, fraction, horizontal_spacing, side)

        for row, col in fig._get_subplot_coordinates():
            subplot = fig.get_subplot(row, col)
            if subplot is None:
                continue
            subplot.xaxis.domain = tuple(scale_x(e, fraction, horizontal_spacing, side) for e in subplot.xaxis.domain)
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
            (n_rows + row, col)
            for row, col in fig._get_subplot_coordinates()
        )
        if side == "top":
            subplot_coordinates2, subplot_coordinates1 = subplot_coordinates2, subplot_coordinates1
    else:
        subplot_coordinates2 = fig._get_subplot_coordinates()
        subplot_coordinates1 = (
            (row, n_cols + col)
            for row, col in fig._get_subplot_coordinates()
        )
        if side == "left":
            subplot_coordinates2, subplot_coordinates1 = subplot_coordinates2, subplot_coordinates1

    new_fig_max_subplot_ids = get_max_subplot_ids(new_fig)

    for (row1, col1), (row2, col2) in zip(subplot_coordinates1, subplot_coordinates2):
        _copy_subplot_ref(new_fig, fig, row1, col1, row2, col2, new_fig_max_subplot_ids)
    # print(new_fig.layout)
    if side in ("top", "bottom"):
        assert side == "bottom"
        for row, col in fig._get_subplot_coordinates():
            add_old_trace_to_new_fig(
                fig, new_fig, new_fig_max_subplot_ids,
                row=row, col=col,
                new_row=n_rows + row, new_col=col
            )
    else:
        for row, col in fig._get_subplot_coordinates():
            add_old_trace_to_new_fig(
                fig, new_fig, new_fig_max_subplot_ids,
                row=row, col=col,
                new_row=row, new_col=n_cols + col
            )

    return new_fig


# def extend_subplot(fig: go.Figure, n_subplots, side="bottom", fraction=0.5,
#                    subplot_titles=None,
#                    vertical_spacing=None, horizontal_spacing=None):
#     n_rows, n_cols = _get_subplot_shape(fig)
#
#     if side in ("top", "bottom"):
#         if n_subplots > n_cols:
#             raise NotImplementedError(f"{n_subplots} > {n_cols}")
#
#         if vertical_spacing is None:
#             vertical_spacing = default_total_vertical_spacing / 2
#         if horizontal_spacing is None:
#             horizontal_spacing = default_total_horizontal_spacing / n_subplots
#
#         _scale_all_objects(fig, side, fraction, vertical_spacing)
#
#         edges = np.linspace(0, 1 - horizontal_spacing * (n_subplots - 1), n_subplots + 1)
#         left_edges = edges[:-1] + np.arange(len(edges[:-1])) * horizontal_spacing
#         right_edges = edges[1:] + np.arange(len(edges[1:])) * horizontal_spacing
#         new_x_domains = zip(left_edges, right_edges)
#         new_y_domains = itertools.repeat((0, fraction - vertical_spacing / 2))
#
#         new_grid_ref_row = [
#             plotly.subplots._init_subplot_xy(fig.layout, False, x_domain, y_domain, get_max_subplot_ids(fig))
#             for x_domain, y_domain in zip(new_x_domains, new_y_domains)
#         ] + [None] * (n_cols - n_subplots)
#         fig._grid_ref.append(new_grid_ref_row)
#
#         new_rows = itertools.repeat(n_rows + 1)
#         new_cols = range(1, n_subplots + 1)
#     elif side in ("right", "left"):
#         if n_subplots > n_rows:
#             raise NotImplementedError(f"{n_subplots} > {n_rows}")
#
#         if vertical_spacing is None:
#             vertical_spacing = default_total_vertical_spacing / n_subplots
#         if horizontal_spacing is None:
#             horizontal_spacing = default_total_horizontal_spacing / 2
#
#         _scale_all_objects(fig, side, fraction, horizontal_spacing)
#
#         edges = np.linspace(0, 1 - vertical_spacing * (n_subplots - 1), n_subplots + 1)
#         left_edges = edges[:-1] + np.arange(len(edges[:-1])) * vertical_spacing
#         right_edges = edges[1:] + np.arange(len(edges[1:])) * vertical_spacing
#         new_x_domains = itertools.repeat((fraction + horizontal_spacing / 2, 1))
#         new_y_domains = zip(left_edges, right_edges)
#
#         new_grid_ref_col = [
#             plotly.subplots._init_subplot_xy(fig.layout, False, x_domain, y_domain, get_max_subplot_ids(fig))
#             for x_domain, y_domain in zip(new_x_domains, new_y_domains)
#         ] + [None] * (n_rows - n_subplots)
#         for i, new_row in enumerate(new_grid_ref_col):
#             fig._grid_ref[i].append(new_row)
#
#         new_rows = itertools.repeat(n_rows + 1)
#         new_cols = range(1, n_subplots + 1)
#     else:
#         raise NotImplementedError(side)
#
#     if subplot_titles is not None:
#         for row, col, title in zip(new_rows, new_cols, subplot_titles):
#             add_subplot_title(fig, title, row, col)
#
#     def _grid_str():
#         raise NotImplementedError
#     fig._grid_str = _grid_str
#
#     return fig


def get_max_subplot_ids(fig):
    return {
        "xaxis": max(
            1 if kw == "xaxis" else int(kw[5:]) for kw in dir(fig.layout) if kw.startswith("xaxis")
        ),
        "yaxis": max(
            1 if kw == "yaxis" else int(kw[5:]) for kw in dir(fig.layout) if kw.startswith("yaxis")
        )
    }


def add_old_trace_to_new_fig(old_fig, new_fig, new_fig_max_subplot_ids, row, col, new_row, new_col):
    from .. import get_traces_at

    traces = get_traces_at(old_fig, row=row, col=col)
    new_fig.add_traces(
        traces,
        rows=[new_row] * len(traces), cols=[new_col] * len(traces)
    )

    # Axes
    if old_fig._grid_ref[row - 1][col - 1] is None:
        return

    subplot_ref = old_fig._grid_ref[row - 1][col - 1][0]
    new_subplot_ref = new_fig._grid_ref[new_row - 1][new_col - 1][0]

    for axis, new_axis in zip(subplot_ref.layout_keys, new_subplot_ref.layout_keys):
        assert axis[0] == new_axis[0]
        axis_dict = {k: v for k, v in old_fig.layout[axis].to_plotly_json().items() if k != "domain"}

        for k in ("anchor", "matches", "scaleanchor", "overlaying"):
            if k in axis_dict.keys():
                axis_type = axis_dict[k][:1]
                axis_id = 1 if len(axis_dict[k]) == 1 else int(axis_dict[k][1:])
                axis_dict[k] = f"{axis_type}{new_fig_max_subplot_ids[f'{axis_type}axis'] + axis_id}"

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

            if (matched_xref := ref_pattern.match(obj.xref)) or ref_pattern.match(obj.yref):
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


def get_new_fit_results(new_fig, fit_results, side):
    new_fit_results = np.ma.empty(
        (*_get_subplot_shape(new_fig), *fit_results.shape[2:]),
        dtype=fit_results.dtype
    )
    new_fit_results.mask = True
    if side == "top":
        nfr = new_fit_results[:fit_results.shape[0]]
    elif side == "bottom":
        nfr = new_fit_results[-fit_results.shape[0]:]
    elif side == "left":
        nfr = new_fit_results[:, :fit_results.shape[1]]
    elif side == "right":
        nfr = new_fit_results[:, -fit_results.shape[1]:]
    else:
        assert False

    sel = np.arange(nfr.size).reshape(nfr.shape) < np.prod(fit_results.shape)

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
        # for shape in fig.select_shapes():
            new_fig.add_shape(copy(shape))

        # Fit Results
        if hasattr(fig, "_fit_results"):
            if not hasattr(new_fig, "_fit_results"):
                new_fig._fit_results = dict()

            for new_fit_name in list(new_fig._fit_results):
                new_fit_results = new_fig._fit_results.pop(new_fit_name)
                new_fig._fit_results[new_fit_name] = get_new_fit_results(new_fig, new_fit_results, "top")

            for fit_name, fit_results in fig._fit_results.items():
                new_fit_results = get_new_fit_results(new_fig, fit_results, "bottom")
                if fit_name in new_fig._fit_results:
                    new_fig._fit_results[fit_name][-fit_results.shape[0]:] = new_fit_results[-fit_results.shape[0]:]
                else:
                    new_fig._fit_results[fit_name] = new_fit_results

    return new_fig


def hstack(fig, *other_figs, fraction=0.5, horizontal_spacing=None):
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
            horizontal_spacing = 2 * horizontal_spacing / (horizontal_spacing + 2) * fraction

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
                new_fig._fit_results[new_fit_name] = get_new_fit_results(new_fig, new_fit_results, "left")

            for fit_name, fit_results in fig._fit_results.items():
                new_fit_results = get_new_fit_results(new_fig, fit_results, "right")
                if fit_name in new_fig._fit_results:
                    new_fig._fit_results[fit_name][:, -fit_results.shape[0]:] = new_fit_results[:, -fit_results.shape[0]:]
                else:
                    new_fig._fit_results[fit_name] = new_fit_results

    return new_fig


def get_subplot_coordinates(
        fig, x_order="left to right", y_order="top to bottom", flatten=True, mask_empty_subplots=False
):
    n_rows, n_cols, *_ = np.array(fig._grid_ref, dtype="O").shape
    a = np.array([
        (row, col)
        for row, col, *_ in sorted(
            (
                (row, col, np.mean(subplot.xaxis.domain), np.mean(subplot.yaxis.domain))
                if subplot is not None
                else (row, col, np.nan, np.nan)
                for row, col, subplot in (
                    (row, col, fig.get_subplot(row, col))
                    for row, col in fig._get_subplot_coordinates()
                )
            ),
            key=lambda row: (1 - row[-1], row[-2])
        )
    ], dtype=[("row", "i8"), ("col", "i8")]).reshape(n_rows, n_cols)

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
        a.mask = np.array([
            next(fig.select_traces(row=row, col=col), None) is None for row, col in a.flatten()
        ]).reshape(a.shape)

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
            obj.select_traces(selector=dict(type="bar")), copied.select_traces(selector=dict(type="bar"))
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
            data = list(fig.select_traces(selector=dict(legendgroup=legendgroup, showlegend=False)))
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
        fig, x_order="left to right", y_order="bottom to top",
        flatten=False, mask_empty_subplots=True
    )
    sel = ~a.mask.view(("?", 2))[..., 0]
    np.put_along_axis(sel, np.argmax(sel, axis=0)[np.newaxis], False, axis=0)
    for row, col in a[sel]:
        fig.update_xaxes(**kwargs, row=row, col=col)


def update_yaxes(fig, target="inside", **kwargs):
    a = get_subplot_coordinates(
        fig, x_order="left to right", y_order="bottom to top",
        flatten=False, mask_empty_subplots=True
    )
    sel = ~a.mask.view(("?", 2))[..., 0]
    np.put_along_axis(sel, np.argmax(sel, axis=1)[:, np.newaxis], False, axis=1)
    for row, col in a[sel]:
        fig.update_yaxes(**kwargs, row=row, col=col)
