import numpy as np
import numpy_utility as npu
import warnings
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
# import plotly_utility
import copy


__all__ = ["histogram", "make_histograms_with_facet_col"]


possible_marginal_types = ["", "rug", "box", "violin", "histogram"]


def _build_dataframe(args):
    if args["weight"] is not None:
        if isinstance(args["weight"], str):
            weight = args["data_frame"][args["weight"]]
        elif npu.is_array(args["weight"]):
            weight = args["weight"]
        args = px._core.build_dataframe(args, go.Histogram)
        assert "weight" not in args["data_frame"].columns
        args["data_frame"]["weight"] = weight
    else:
        args = px._core.build_dataframe(args, go.Histogram)
        assert "weight" not in args["data_frame"].columns
    return args


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
    category_orders={},
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
    as_qualitative=False,
    log_bin_x=False
) -> go.Figure:
    """
    Precomputing histogram binning in Python, not in Javascript.
    """
    local = locals()
    if len(labels) == 0:
        args = _build_dataframe(local)
        args["labels"] = {}  # Prevent that labels is set automatically if marginal=="rug"
    else:
        args = _build_dataframe(local)
    return _histogram(args, log_bin_x)


def _histogram(args, log_bin_x):
    if args["marginal"] is None:
        pass
    elif args["marginal"] not in possible_marginal_types:
        raise ValueError(f"""
    Got invalid marginal '{args['marginal']}'.
    Possible marginal: {", ".join(possible_marginal_types)}
        """)

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

    if args["as_qualitative"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            args["data_frame"].loc[:, args["x"]] = args["data_frame"].loc[:, args["x"]].astype(str)

    # args["data_frame"] = args["data_frame"].dropna(subset=[args["x"]])
    data = args["data_frame"][args["x"]].to_numpy()
    if npu.is_numeric(data):
        is_category = False
        sel = np.isfinite(data)
        args["data_frame"] = args["data_frame"][sel]
        data = args["data_frame"][args["x"]].to_numpy()
    else:
        is_category = True

    if np.issubdtype(data.dtype, np.object_):
        data = np.array(data.tolist())

    weight = args["data_frame"]["weight"] if "weight" in args["data_frame"] else None

    if np.issubdtype(data.dtype, np.datetime64):
        x_converted = data.astype("M8[us]")
        if x_converted.size != data.size:
            warnings.warn(f"""
    histogram does not draw accurately using datetime objects with more precise unit than [us]:
        the maximum number of bins: {len(np.unique(data))} -> {len(np.unique(x_converted))}
    For more accurate histogram, You should use px.histogram or lose the precise somehow.
            """)
        data = x_converted

    bins = npu.histogram_bin_edges(data, bins=bins, weights=weight, log=log_bin_x)
    bin_width = npu.histogram_bin_widths(bins)
    x = npu.histogram_bin_centers(bins)

    if len(data) == 0:
        return px.bar()

    use_one_plot = (args["color"] is None) and (args["facet_row"] is None) and (args["facet_col"] is None)

    if args["histnorm"] is None or args["histnorm"] == "":
        density = False
    elif args["histnorm"] == "probability density":
        density = True
    elif args["histnorm"] == "probability":
        density = False
    elif args["histnorm"] == "percent":
        density = False
    else:
        raise NotImplementedError(f"histnorm={args['histnorm']} not supported yet")

    if use_one_plot:
        y, _ = npu.histogram(data, bins=bins, density=density, weights=weight)
        data_classes = np.array([""] * len(data))
        args["data_frame"] = pd.DataFrame()
    else:
        groups = []
        if args["color"] is not None:
            groups.append(args["data_frame"][args["color"]])
        if args["facet_row"] is not None:
            groups.append(args["data_frame"][args["facet_row"]])
        if args["facet_col"] is not None:
            groups.append(args["data_frame"][args["facet_col"]])

        data_classes = np.array(np.transpose(groups).tolist())
        un = np.unique(data_classes, axis=0)

        y = np.array([
            y
            for d, w in (
                (data[sel], weight[sel] if weight is not None else None)
                for sel in (np.all(data_classes == n, axis=1) for n in un)
            )
            for y in npu.histogram(d, bins=bins, density=density, weights=w)[0]
        ])

        groups = np.array([
            name
            for name in un
            for _ in x
        ]).T.tolist()

        args["data_frame"] = pd.DataFrame()

        if args["facet_col"] is not None:
            args["data_frame"][args["facet_col"]] = groups.pop()
        if args["facet_row"] is not None:
            args["data_frame"][args["facet_row"]] = groups.pop()
        if args["color"] is not None:
            args["data_frame"][args["color"]] = groups.pop()

        assert y.size % x.size == 0
        _n_unique_names = y.size // x.size
        x = np.tile(x, _n_unique_names)
        # bin_width = np.tile(bin_width, _n_unique_names)

    args["data_frame"][args["x"]] = x
    args["data_frame"][args["y"]] = y

    if swap_xy:
        args["x"], args["y"] = args["y"], args["x"]
        args["orientation"] = "h"
        y_axis = "x"
    else:
        y_axis = "y"

    assert y_axis not in args["labels"]
    if args["histnorm"] == "probability density":
        y_label = "density"
    elif args["histnorm"] == "probability":
        y_label = "probability"
        args["data_frame"][args["y"]] /= args["data_frame"][args["y"]].sum()
    elif args["histnorm"] == "percent":
        y_label = "percent"
        args["data_frame"][args["y"]] /= args["data_frame"][args["y"]].sum()
        args["data_frame"][args["y"]] *= 100
    else:
        y_label = "count"

    args["labels"].update({y_axis: y_label})

    fig = px._core.make_figure(
        args=args,
        constructor=go.Bar,
        trace_patch=dict(
            textposition="auto",
            width=bin_width // 1000 if np.issubdtype(data.dtype, np.datetime64) else bin_width,
            marker_line_width=0,
            orientation=args["orientation"]
        ),
        layout_patch=dict(
            barmode=args["barmode"],
            bargap=0
        )
    )

    if is_category:
        fig.update_xaxes(type="category")

    has_marginal = args["marginal"] is not None and args["marginal"] != ""
    if has_marginal:
        # marginal_traces = plotly_utility.get_traces_at(fig, 2, "all")
        marginal_traces = [t for t in fig.data if t.type in possible_marginal_types]
        if use_one_plot:
            assert len(marginal_traces) == 1
            marginal_traces[0].x = data
        else:
            assert len(marginal_traces) == len(un)
            for trace, n in zip(marginal_traces, un):
                # Replace the binned marginal plots to the unbinned ones
                trace.x = data[np.all(data_classes == n, axis=1)]

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

