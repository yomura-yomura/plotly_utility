import functools
import plotly.express as px
from . import _histogram
import plotly.graph_objs as go


__all__ = ["scatter"]


@functools.wraps(px.scatter)
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
        category_orders={},
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
):
    args = px._core.build_dataframe(locals(), go.Scatter)
    fig = px._core.make_figure(args=args, constructor=go.Scatter)

    data_frame = args["data_frame"]
    x = args["x"]
    y = args["y"]
    color = args["color"]

    # colored_traces = [trace for trace in fig.data if trace.xaxis == "xaxis" and trace.yaxis == "yaxis"]
    # color = [ct.name for ct in colored_traces]

    copied_data = list(fig.data)
    fig.data = ()

    fig.add_traces([t for t in copied_data if t.xaxis == "x" and t.yaxis == "y"])

    if "histogram" in (marginal_x, marginal_y):
        if marginal_x == "histogram":
            new_marginal_x_histogram = _histogram.histogram(data_frame, x=x, color=color)
            new_marginal_x_histogram.update_traces(opacity=0.5)
            fig.add_traces(new_marginal_x_histogram.data,
                           rows=[2] * len(new_marginal_x_histogram.data),
                           cols=[1] * len(new_marginal_x_histogram.data))

        if marginal_y == "histogram":
            new_marginal_y_histogram = _histogram.histogram(data_frame, y=y, color=color)
            new_marginal_y_histogram.update_traces(opacity=0.5)
            fig.add_traces(new_marginal_y_histogram.data,
                           rows=[1] * len(new_marginal_y_histogram.data),
                           cols=[2] * len(new_marginal_y_histogram.data))

        fig.update_layout(
            # barmode=barmode,
            bargap=0
        )

    return fig
