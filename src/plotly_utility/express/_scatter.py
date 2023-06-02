import plotly.express as px
from . import _histogram
import plotly.graph_objs as go
import numpy_utility as npu
from . import _core


__all__ = ["scatter"]


def scatter(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    symbol=None,
    size=None,
    hover_name=None,
    hover_data=None,
    custom_data=None,
    text=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    error_x=None,
    error_x_minus=None,
    error_y=None,
    error_y_minus=None,
    animation_frame=None,
    animation_group=None,
    category_orders=None,
    labels={},
    orientation=None,
    color_discrete_sequence=None,
    color_discrete_map={},
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    symbol_sequence=None,
    symbol_map={},
    opacity=None,
    size_max=None,
    marginal_x=None,
    marginal_y=None,
    trendline=None,
    trendline_color_override=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    render_mode="auto",
    title=None,
    template=None,
    width=None,
    height=None,

    nbinsx=None, nbinsy=None,
    use_lob_x_bins=False, use_lob_y_bins=False
):

    if npu.is_array(facet_col):
        sep = ", "
        _new_face_col = sep.join(facet_col)
        data_frame[_new_face_col] = data_frame[facet_col].apply(
            lambda s: s.astype(str)
        ).apply(
            lambda r: sep.join(r), axis=1
        )
        facet_col = _new_face_col

    args = _core.build_dataframe(locals(), go.Scatter)
    return _scatter(args)


def _scatter(args):
    fig = px._core.make_figure(args=args, constructor=go.Scatter)

    if args["facet_col"] is None:
        data_frame = args["data_frame"]
        x = args["x"]
        y = args["y"]
        color = args["color"]

        # colored_traces = [trace for trace in fig.data if trace.xaxis == "xaxis" and trace.yaxis == "yaxis"]
        # color = [ct.name for ct in colored_traces]
        # print([(t.name, t.marker.color) for t in fig.data if t.xaxis == "x" and t.yaxis == "y"])
        # copied_data = list(fig.data)
        fig.data = tuple(t for t in fig.data if t.xaxis == "x" and t.yaxis == "y")

        if "histogram" in (args["marginal_x"], args["marginal_y"]):
            if args["marginal_x"] == "histogram":
                new_marginal_x_histogram = _histogram.histogram(
                    data_frame, x=x, color=color, nbins=args["nbinsx"], category_orders=args["category_orders"],
                    use_log_x_bins=args["use_lob_x_bins"]
                )
                new_marginal_x_histogram.update_traces(opacity=0.5, showlegend=False)
                fig.add_traces(new_marginal_x_histogram.data,
                               rows=[2] * len(new_marginal_x_histogram.data),
                               cols=[1] * len(new_marginal_x_histogram.data))

            if args["marginal_y"] == "histogram":
                new_marginal_y_histogram = _histogram.histogram(
                    data_frame, y=y, color=color, nbins=args["nbinsy"], category_orders=args["category_orders"],
                    use_log_x_bins=args["use_lob_y_bins"]
                )
                new_marginal_y_histogram.update_traces(opacity=0.5, showlegend=False)
                fig.add_traces(new_marginal_y_histogram.data,
                               rows=[1] * len(new_marginal_y_histogram.data),
                               cols=[2] * len(new_marginal_y_histogram.data))

            fig.update_layout(
                # barmode=barmode,
                bargap=0
            )

    return fig
