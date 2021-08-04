import numpy as np
import plotly_utility

import numpy_utility as npu
import warnings
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
# import plotly_utility
import copy
from . import _core


__all__ = ["histogram", "make_histograms_with_facet_col"]


possible_marginal_types = ["", "rug", "box", "violin", "histogram"]


def histogram(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    marginal=None,
    opacity=None,
    orientation=None,
    barmode="relative",
    barnorm=None,
    histnorm=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    # histfunc=None,
    # cumulative=None,
    nbins=None,
    title=None,
    template=None,
    width=None,
    height=None,

    weight=None,
    as_qualitative=None,
    use_different_bin_widths=False,
    disable_xaxis_matches=False,
    disable_yaxis_matches=False,
    marginal_residual_plot=False,
    use_log_x_bins=False
) -> go.Figure:
    """
    Precomputing histogram binning in Python, not in Javascript.
    """
    args = _core.build_dataframe(locals(), go.Histogram)
    return _histogram(args)


def normalize(histnorm, y, bin_width):
    if histnorm is None or histnorm == "":
        return y
    elif histnorm == "probability density":
        return y / (y * bin_width).sum()
    elif histnorm == "probability":
        return y / y.sum()
    elif histnorm == "percent":
        return y / y.sum() * 100
    elif histnorm == "density":
        return y / bin_width
    else:
        assert False


def get_y_title(histnorm):
    if histnorm is None or histnorm == "":
        y_title = "count"
    elif histnorm == "probability density":
        y_title = "probability density"
    elif histnorm == "probability":
        y_title = "probability"
    elif histnorm == "percent":
        y_title = "percent"
    elif histnorm == "density":
        y_title = "density"
    else:
        assert False
    return y_title


def _histogram(args):
    if args["marginal"] is None:
        pass
    elif args["marginal"] not in possible_marginal_types:
        raise ValueError(f"""
    Got invalid marginal '{args['marginal']}'.
    Possible marginal: {", ".join(possible_marginal_types)}
        """)

    if args["category_orders"] is None:
        args["category_orders"] = dict()
    else:
        args["category_orders"] = args["category_orders"].copy()

    if args["labels"] is None:
        args["labels"] = dict()
    else:
        args["labels"] = args["labels"].copy()

    if args["histnorm"] not in (None, "", "probability density", "probability", "percent", "density"):
        raise ValueError(f"unexpected value encountered: histnorm={args['histnorm']}")

    if args["nbins"] is None:
        bins = "auto"
    else:
        bins = args["nbins"]

    if (args["x"] is None) and (args["y"] is None):
        return px.bar()
    elif (args["x"] is not None) and (args["y"] is None or args["y"] not in args["data_frame"].columns):
        args["y"] = "y"
        swap_xy = False
    elif (args["x"] is None or args["x"] not in args["data_frame"].columns) and (args["y"] is not None):
        args["x"] = "x"
        args["x"], args["y"] = args["y"], args["x"]
        swap_xy = True
    else:
        raise NotImplementedError("Not supported yet in the case x and y are not None ")

    if args["as_qualitative"] is not None:
        if not npu.is_array(args["as_qualitative"]):
            args["as_qualitative"] = [args["as_qualitative"]]

        for label in args["as_qualitative"]:
            if label not in args["data_frame"].columns:
                raise ValueError(f"'{label}' specified in as_qualitative is not found in df.columns")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sort_by = np.unique(args["data_frame"].loc[:, label]).astype(str)
                args["data_frame"].loc[:, label] = args["data_frame"].loc[:, label].astype(str)
                if label not in args["category_orders"]:
                    if label in args["labels"]:
                        label = args["labels"][label]
                    args["category_orders"][label] = sort_by.tolist()
                else:
                    raise NotImplementedError(f"'{label}' must not be included in category_orders if as_qualitative == True")

    data = args["data_frame"][args["x"]].to_numpy()
    weight = args["data_frame"]["weight"] if "weight" in args["data_frame"] else None

    if np.issubdtype(data.dtype, np.object_):
        data = np.array(data.tolist())

    if npu.is_numeric(data):
        is_category = False
        sel = np.isfinite(data)
        args["data_frame"] = args["data_frame"][sel]
        data = args["data_frame"][args["x"]].to_numpy()
    elif np.issubdtype(data.dtype, np.datetime64):
        is_category = False
        x_converted = data.astype("M8[us]")
    #     if x_converted.size != data.size:
    #         warnings.warn(f"""
    # histogram does not draw accurately using datetime objects with more precise unit than [us]:
    #     the maximum number of bins: {len(np.unique(data))} -> {len(np.unique(x_converted))}
    # For more accurate histogram, You should use px.histogram or lose the precise somehow.
    #         """)
        data = x_converted
    else:
        is_category = True

    if len(data) == 0:
        return px.bar()

    use_one_plot = (args["color"] is None) and (args["facet_row"] is None) and (args["facet_col"] is None)

    density = False

    def iter_over_counts_centers_widths(data, bins, density, weight, use_log_bins):
        bins = npu.histogram_bin_edges(data, bins=bins, weights=weight, log=use_log_bins)
        counts, bins = npu.histogram(data, bins=bins, density=density, weights=weight)
        width = npu.histogram_bin_widths(bins)
        center = npu.histogram_bin_centers(bins)
        assert len(counts) == len(center) == len(width)
        return zip(counts, center, width)

    if use_one_plot:
        assert args["use_different_bin_widths"] == np.False_  # maybe should be removed

        # y, bins = npu.histogram(data, bins=bins, density=density, weights=weight)
        # x = npu.histogram_bin_centers(bins)
        # bin_width = npu.histogram_bin_widths(bins)
        y, x, bin_width = zip(*iter_over_counts_centers_widths(data, bins, density, weight, args["use_log_x_bins"]))

        args["data_frame"] = pd.DataFrame()
        args["data_frame"][args["x"]] = x
        args["data_frame"][args["y"]] = y
        args["data_frame"][args["y"]] = normalize(args["histnorm"], args["data_frame"][args["y"]], bin_width)
    else:
        groups = {}
        if args["facet_col"] is not None:
            groups["facet_col"] = np.array(args["data_frame"][args["facet_col"]].tolist())
        if args["facet_row"] is not None:
            groups["facet_row"] = np.array(args["data_frame"][args["facet_row"]].tolist())
        if args["color"] is not None:
            groups["color"] = np.array(args["data_frame"][args["color"]].tolist())
        groups = npu.from_dict(groups)

        unique_groups = np.unique(groups)

        if args["use_different_bin_widths"] == np.True_:
            pass
        else:
            bins = npu.histogram_bin_edges(data, bins=bins, weights=weight, log=args["use_log_x_bins"])

        tidy_data = np.array([
            (count, center, width, ug, np.nan)
            for ug, sel in ((ug, groups == ug) for ug in unique_groups)
            for count, center, width in iter_over_counts_centers_widths(
                data[sel], bins, density, weight[sel] if weight is not None else None, args["use_log_x_bins"]
            )
        ], dtype=[("y", "f8"), ("x", data.dtype), ("bin_width", "f8"), ("group", groups.dtype), ("error_y", "f8")])

        for ug in np.unique(tidy_data["group"]):
            tidy_data["y"][tidy_data["group"] == ug] = normalize(
                args["histnorm"],
                tidy_data["y"][tidy_data["group"] == ug],
                tidy_data["bin_width"][tidy_data["group"] == ug]
            )

        bin_width = tidy_data["bin_width"]

        args["data_frame"] = pd.DataFrame()
        args["data_frame"][args["x"]] = tidy_data["x"]
        args["data_frame"][args["y"]] = tidy_data["y"]

        if args["facet_col"] is not None:
            args["data_frame"][args["facet_col"]] = tidy_data["group"]["facet_col"]
        if args["facet_row"] is not None:
            args["data_frame"][args["facet_row"]] = tidy_data["group"]["facet_row"]
        if args["color"] is not None:
            args["data_frame"][args["color"]] = tidy_data["group"]["color"]

    # swap_xy after

    if swap_xy:
        args["x"], args["y"] = args["y"], args["x"]
        args["orientation"] = "h"
        y_axis = "x"
    else:
        y_axis = "y"

    assert y_axis not in args["labels"]
    y_label = get_y_title(args["histnorm"])
    args["labels"].update({y_axis: y_label})

    if args["barmode"] == "group":
        bargap = None
        bin_width = None
    else:
        bargap = 0

    fig = px._core.make_figure(
        args=args,
        constructor=go.Bar,
        trace_patch=dict(
            textposition="auto",
            # width=bin_width // 1000 if np.issubdtype(data.dtype, np.datetime64) else bin_width,
            marker_line_width=0,
            orientation=args["orientation"]
        ),
        layout_patch=dict(
            barmode=args["barmode"],
            bargap=bargap,
            bargroupgap=0
        )
    )

    if args["disable_xaxis_matches"] == np.True_:
        fig.update_xaxes(showticklabels=True, matches=None)
    if args["disable_yaxis_matches"] == np.True_:
        fig.update_yaxes(showticklabels=True, matches=None)

    if is_category:
        fig.update_xaxes(type="category")

    # Bin Widths
    if bin_width is not None:
        bin_width = bin_width // 1000 if np.issubdtype(data.dtype, np.datetime64) else bin_width

    if use_one_plot:
        fig.data[0].width = bin_width
        fig.data[0]._x = data
    else:
        if args["facet_col"] in args["labels"]:
            args["facet_col"] = args["labels"][args["facet_col"]]
        if args["facet_row"] in args["labels"]:
            args["facet_row"] = args["labels"][args["facet_row"]]
        if args["color"] in args["labels"]:
            args["color"] = args["labels"][args["color"]]

        for trace in fig.data:
            if trace.type == "bar":
                import re

                trace_id = []
                
                if args["facet_col"] is not None:
                    matched = re.findall(rf"(?:\A|<br>){args['facet_col']}=(.+?)<br>", trace.hovertemplate)
                    assert len(matched) == 1
                    trace_id.append(matched[0])
                if args["facet_row"] is not None:
                    matched = re.findall(rf"(?:\A|<br>){args['facet_row']}=(.+?)<br>", trace.hovertemplate)
                    assert len(matched) == 1
                    trace_id.append(matched[0])
                if args["color"] is not None:
                    matched = re.findall(rf"(?:\A|<br>){args['color']}=(.+?)<br>", trace.hovertemplate)
                    assert len(matched) == 1
                    trace_id.append(matched[0])
                key = np.array(tuple(trace_id), tidy_data["group"].dtype)

                trace._x = data[groups == key]

                if bin_width is not None:
                    trace.width = bin_width[tidy_data["group"] == key].tolist()

    # Marginal Plots

    has_marginal = args["marginal"] is not None and args["marginal"] != ""
    if has_marginal:
        marginal_traces = [trace for trace in fig.data if trace.type in possible_marginal_types]
        if use_one_plot:
            assert len(marginal_traces) == 1
            marginal_traces[0].x = data
        else:
            assert len(marginal_traces) == len(unique_groups)
            for trace in marginal_traces:
                # Replace the binned marginal plots to the unbinned ones
                trace.x = data[groups["color"] == trace.name]

    return fig


def make_histograms_with_facet_col(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    animation_frame=None,
    animation_group=None,
    # category_orders={},
    labels={},
    color_discrete_sequence=None,
    color_discrete_map={},
    marginal=None,
    opacity=None,
    orientation=None,
    barmode="relative",
    barnorm=None,
    histnorm=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    # histfunc=None,
    # cumulative=None,
    nbins=None,
    title=None,
    template=None,
    width=None,
    height=None,

    weight=None,
    as_qualitative=False
):
    args = _build_dataframe(locals())
    return _make_histograms_with_facet_col(args)


def _make_histograms_with_facet_col(args):
    if args["facet_col"] is None:
        raise ValueError(f"facet_col argument must be specified")

    df = args["data_frame"]
    unique_seperater, indices = np.unique(df[args["facet_col"]], return_index=True)
    unique_seperater = unique_seperater[indices.argsort()]

    n_pages = unique_seperater.size // 50 + 1
    for sep_at_page in np.array_split(unique_seperater, n_pages):
        args["data_frame"] = df[np.isin(df[args["facet_col"]], sep_at_page)]
        args["facet_col_wrap"] = int(np.ceil(np.sqrt(len(sep_at_page))))
        args["category_orders"] = {"facet_col": sep_at_page.tolist()}
        yield _histogram(copy.deepcopy(args))

