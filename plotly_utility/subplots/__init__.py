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

    sel = (
        (titles_x[..., np.newaxis] == annotations_x[np.newaxis, np.newaxis, :]) &
        (titles_y[..., np.newaxis] == annotations_y[np.newaxis, np.newaxis, :])
    )  # (row, col, annotation)

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


def _copy_subplot_ref(new_fig, fig, new_row, new_col, row, col):
    new_i_row = new_row - 1
    new_i_col = new_col - 1
    i_row = row - 1
    i_col = col - 1

    subplot_ref = fig._grid_ref[i_row][i_col][0]
    new_subplot_ref = plotly.subplots._init_subplot_xy(
        new_fig.layout, False, *(fig.layout[kw]["domain"] for kw in subplot_ref.layout_keys),
        get_max_subplot_ids(new_fig)
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
        for annotation in fig.layout.annotations:
            if annotation.yref == "paper":
                annotation.y = scale_y(annotation.y, fraction, vertical_spacing, side)
        for image in fig.layout.images:
            if image.yref == "paper":
                image.y = scale_y(image.y, fraction, vertical_spacing, side)
                image.sizey = scale_y(image.sizey, fraction, vertical_spacing, "top")

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
        for annotation in fig.layout.annotations:
            if annotation.xref == "paper":
                annotation.x = scale_x(annotation.x, fraction, horizontal_spacing, side)
        for image in fig.layout.images:
            if image.xref == "paper":
                image.x = scale_x(image.x, fraction, horizontal_spacing, side)
                image.sizex = scale_x(image.sizex, fraction, horizontal_spacing, "right")

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


def combine_subplots(fig1, fig2, side, fraction=0.5, spacing=None):
    if side not in ("top", "bottom", "right", "left"):
        raise ValueError("Available side: top, bottom, right or left")

    if side in ("top", "left"):
        fig1, fig2 = fig1, fig2

    fig1 = copy_figure(fig1)
    _scale_all_objects(fig1, side, fraction, spacing)
    fig2 = copy_figure(fig2)
    _scale_all_objects(fig2, _get_opposite_side(side), fraction, spacing)
    n_rows, n_cols = _get_subplot_shape(fig1)

    if side in ("top", "bottom"):
        subplot_coordinates = fig1._get_subplot_coordinates()
        new_subplot_coordinates = (
            (n_rows + row, col)
            for row, col in fig2._get_subplot_coordinates()
        )
        if side == "top":
            new_subplot_coordinates, subplot_coordinates = new_subplot_coordinates, subplot_coordinates
    else:
        subplot_coordinates = fig1._get_subplot_coordinates()
        new_subplot_coordinates = (
            (row, n_cols + col)
            for row, col in fig2._get_subplot_coordinates()
        )
        if side == "left":
            new_subplot_coordinates, subplot_coordinates = new_subplot_coordinates, subplot_coordinates

    for (new_row, new_col), (row, col) in zip(new_subplot_coordinates, subplot_coordinates):
        if fig2._grid_ref[row-1][col-1] is None:
            continue
        _copy_subplot_ref(fig1, fig2, new_row, new_col, row, col)
    return fig1


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

        _scale_all_objects(fig, side, fraction, vertical_spacing)

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

        _scale_all_objects(fig, side, fraction, horizontal_spacing)

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


def add_old_trace_to_new_fig(fig, new_fig, row, col, new_row, new_col, hold_domain_of=None):
    assert hold_domain_of in ("x", "y", None)

    from .. import get_traces_at
    # new_max_subplot_ids = get_max_subplot_ids(new_fig)

    traces = get_traces_at(fig, row=row, col=col)
    new_fig.add_traces(
        traces,
        rows=[new_row] * len(traces), cols=[new_col] * len(traces)
    )

    # Axes
    if fig._grid_ref[row-1][col-1] is None:
        return

    subplot_ref = fig._grid_ref[row-1][col-1][0]
    new_subplot_ref = new_fig._grid_ref[new_row - 1][new_col - 1][0]

    new_subplot_axes_ids = {
        k: (1 if v in ("x", "y") else int(v[1:]))
        for k, v in new_subplot_ref.trace_kwargs.items()
    }
    # import pprint
    # pprint.pprint(new_fig._grid_ref)
    # print(new_subplot_axes_ids)
    # print({
    #     k: new_max_subplot_ids[k] + (1 if len(v) == 1 else int(v[1:])) - 1
    #     for k, v in subplot_ref.trace_kwargs.items()
    # })
    # assert new_subplot_axes_ids == {
    #     k: new_max_subplot_ids[k] + (1 if len(v) == 1 else int(v[1:])) - 1
    #     for k, v in subplot_ref.trace_kwargs.items()
    # }

    def old_to_new_axis_id(axis_id: str):
        axis_type = axis_id[:1]
        # new_id = (int(axis_id[1:]) if axis_id[1:] != "" else 1) + new_max_subplot_ids[f"{axis_type}axis"] - 1
        new_id = new_subplot_axes_ids[f"{axis_type}axis"]
        if new_id == 1:
            return axis_type
        else:
            return f"{axis_type}{new_id}"

    for axis, new_axis in zip(subplot_ref.layout_keys, new_subplot_ref.layout_keys):
        assert axis[0] == new_axis[0]
        # axis_dict = {k: v for k, v in fig.layout[axis].to_plotly_json().items()
        #              if hold_domain_of is None or axis.startswith(hold_domain_of) or k not in ("domain",)}
        axis_dict = {k: v for k, v in fig.layout[axis].to_plotly_json().items() if k != "domain"}
        for k in ("anchor", "matches", "scaleanchor", "overlaying"):
            if k in axis_dict.keys():
                axis_dict[k] = old_to_new_axis_id(axis_dict[k])
        new_fig.layout[new_axis].update(axis_dict)


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


def vstack(fig, other_fig, fraction=0.5, vertical_spacing=None):
    new_fig = copy_figure(fig)
    figs = [other_fig]
    assert len(figs) == 1

    new_fig_shape = _get_subplot_shape(new_fig)

    if vertical_spacing is None:
        vertical_spacing = default_total_vertical_spacing / 2

    for fig in figs:
        n_rows, n_cols = _get_subplot_shape(fig)
        # subplot_titles = get_subplot_titles(fig)
        # assert n_rows == 1

        new_fig = combine_subplots(new_fig, fig, "bottom", fraction, vertical_spacing)

        for i_row in range(n_rows):
            for i_col in range(n_cols):
                add_old_trace_to_new_fig(
                    fig, new_fig,
                    row=i_row + 1, col=i_col + 1,
                    new_row=new_fig_shape[0] + i_row + 1, new_col=i_col + 1,
                    # hold_domain_of="x"
                )

        fig = copy_figure(fig)
        _scale_all_objects(fig, "top", fraction, vertical_spacing)

        # Images
        for image in fig.layout.images:
            new_image = go.layout.Image(image)
            assert new_image.yref == "paper"
            new_fig.add_layout_image(new_image)

        # Annotations
        for annotation in fig.layout.annotations:
            new_annotation = go.layout.Annotation(annotation)
            assert new_annotation.yref == "paper"
            new_fig.add_annotation(new_annotation)

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

    # assert all(_get_subplot_shape(new_fig) == fr.shape[:2] for fr in new_fig._fit_results.values())

    return new_fig


def hstack(fig, other_fig, fraction=0.5, horizontal_spacing=None):
    new_fig = copy_figure(fig)
    figs = [other_fig]
    assert len(figs) == 1

    new_fig_shape = _get_subplot_shape(new_fig)

    if horizontal_spacing is None:
        horizontal_spacing = default_total_horizontal_spacing / 2

    for fig in figs:
        n_rows, n_cols = _get_subplot_shape(fig)
        # subplot_titles = get_subplot_titles(fig)
        # assert n_cols == 1

        new_fig = combine_subplots(new_fig, fig, "right", fraction, horizontal_spacing)

        for i_col in range(n_cols):
            for i_row in range(n_rows):
                add_old_trace_to_new_fig(
                    fig, new_fig,
                    row=i_row + 1, col=i_col + 1,
                    new_row=i_row + 1, new_col=new_fig_shape[1] + i_col + 1,
                    hold_domain_of="y"
                )

        fig = copy_figure(fig)
        _scale_all_objects(fig, "left", fraction, horizontal_spacing)

        # Images
        for image in fig.layout.images:
            new_image = go.layout.Image(image)
            assert new_image.xref == "paper"
            new_fig.add_layout_image(new_image)

        # Annotations
        for annotation in fig.layout.annotations:
            new_annotation = go.layout.Annotation(annotation)
            assert new_annotation.xref == "paper"
            new_fig.add_annotation(new_annotation)

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

    # assert all(_get_subplot_shape(new_fig) == fr.shape[:2] for fr in new_fig._fit_results.values())

    return new_fig


def get_subplot_coordinates(fig, x_order="left to right", y_order="top to bottom"):
    return (
        (row, col)
        for row, col, *_ in sorted(
            (
                (row, col, np.mean(subplot.xaxis.domain), np.mean(subplot.yaxis.domain))
                for row, col, subplot in (
                    (row, col, fig.get_subplot(row, col))
                    for row, col in fig._get_subplot_coordinates()
                )
            ),
            key=lambda row: (1 - row[-1], row[-2])
        )
    )


def copy_figure(fig):
    import copy
    copied = go.Figure(fig)
    copied._grid_ref = copy.deepcopy(copied._grid_ref)
    if hasattr(fig, "_fit_results"):
        copied._fit_results = copy.deepcopy(fig._fit_results)
    return copied
