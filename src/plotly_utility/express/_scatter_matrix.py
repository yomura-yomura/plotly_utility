import io
import itertools
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots

from ..mpl_utils import mpl_batch_mode
from . import _histogram

__all__ = ["scatter_matrix"]


def scatter_matrix(df, color=None, barmode=None, width=None, height=None):
    df = pd.DataFrame(df)

    if color is not None:
        _color = df[color]
        df = df.drop(columns=color)
        color = _color

    n_cols = n_rows = len(df.columns)

    fig = plotly.subplots.make_subplots(n_rows, n_cols)

    for i_row, i_col in itertools.product(range(n_rows), range(n_cols)):
        if i_row >= i_col:
            if i_row == i_col:
                _fig = _histogram.histogram(
                    x=df.iloc[:, i_row],
                    color=color,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    barmode=barmode,
                )
                if color is None:
                    _fig = _fig.update_traces(name=f"dist plot of {df.columns[i_row]}")
                traces = _fig.data

                fig.add_traces(
                    traces,
                    rows=[i_row + 1] * len(traces),
                    cols=[i_col + 1] * len(traces),
                )

            elif i_row > i_col:
                _fig = px.scatter(
                    # mode="markers",
                    x=df.iloc[:, i_col],
                    y=df.iloc[:, i_row],
                    color=color,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                if color is None:
                    _fig = _fig.update_traces(
                        name=f"scatter plot of {df.columns[i_row]} vs {df.columns[i_col]}"
                    )
                traces = _fig.data

                fig.add_traces(
                    traces,
                    rows=[i_row + 1] * len(traces),
                    cols=[i_col + 1] * len(traces),
                )

                fig.update_yaxes(
                    matches=fig.get_subplot(i_row + 1, 1).xaxis.anchor,
                    showticklabels=False,
                    row=i_row + 1,
                    col=i_col + 1,
                )
            fig.update_xaxes(
                matches=fig.get_subplot(n_rows, i_col + 1).yaxis.anchor,
                showticklabels=False,
                row=i_row + 1,
                col=i_col + 1,
            )
        else:
            subplot = fig.get_subplot(i_row + 1, i_col + 1)

            if color is not None:
                warnings.warn(
                    "multiple corr plots not supported yet if color is not None",
                    UserWarning,
                )

            with mpl_batch_mode():
                try:
                    plt.figure(figsize=(2, 2))
                    _corrdot(df.iloc[:, i_row], df.iloc[:, i_col])

                    with io.BytesIO() as fp:
                        plt.savefig(fp, format="png")
                        img = PIL.Image.open(fp)

                        fig.add_layout_image(
                            xref="x domain",
                            yref="y domain",
                            row=i_row + 1,
                            col=i_col + 1,
                            # xref="paper", yref="paper",
                            xanchor="center",
                            yanchor="middle",
                            x=np.mean(subplot.xaxis.domain),
                            y=np.mean(subplot.yaxis.domain),
                            sizex=subplot.xaxis.domain[1] - subplot.xaxis.domain[0],
                            sizey=subplot.yaxis.domain[1] - subplot.yaxis.domain[0],
                            source=img,
                        )
                except TypeError as e:
                    print(f"encountered {e.__class__.__name__}: {e}", file=sys.stderr)
                plt.close()

    for i in range(n_rows):
        fig.update_xaxes(
            title=df.columns[i], showticklabels=True, row=n_rows, col=i + 1
        )
        fig.update_yaxes(title=df.columns[i], showticklabels=True, row=i + 1, col=1)

    fig.update_traces(showlegend=False)
    fig.update_layout(bargap=0, width=width, height=height)

    return fig


def _corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], "pearson")
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")

    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter(
        [0.5],
        [0.5],
        marker_size,
        [corr_r],
        alpha=0.6,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        transform=ax.transAxes,
    )
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(
        corr_text,
        [
            0.5,
            0.5,
        ],
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=font_size,
    )


# def _scatter_matrix(df, filename=None):
#     import seaborn as sns
#
#     df = pd.DataFrame(df)
#
#     sns.set(style='white', font_scale=1.6)
#     g = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
#     g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
#     g.map_diag(sns.histplot, kde_kws={'color': 'black'})
#     g.map_upper(_corrdot)
#
#     if filename is not None:
#         plt.savefig(filename)
#         # import webbrowser
#         # webbrowser.open_new(r"file://C:{}".format(str(filename).replace("/", "\\")))
#     plt.show()
#     # plt.draw()
#     # plt.pause(0.001)
