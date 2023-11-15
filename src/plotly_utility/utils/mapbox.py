import pathlib

import numpy as np
import plotly.graph_objs as go

__all__ = ["init_mapbox_fig", "set_auto_margin"]


def init_mapbox_fig(fig: go.Figure):
    fig.layout.mapbox.accesstoken = _load_api_key()
    set_auto_margin(fig)


def set_auto_margin(
    fig: go.Figure,
    margin_scale=1.2,
    graph_width_scale_in_fig=0.9,
    graph_height_scale_in_fig=0.5,
):
    x1 = min(lon for trace in fig.data for lon in trace["lon"])
    x2 = max(lon for trace in fig.data for lon in trace["lon"])
    y1 = min(lat for trace in fig.data for lat in trace["lat"])
    y2 = max(lat for trace in fig.data for lat in trace["lat"])

    width = fig.layout.width * graph_width_scale_in_fig
    height = fig.layout.height * graph_height_scale_in_fig
    max_bound = max(
        (margin_scale * abs(x1 - x2) * 111) / width,
        (margin_scale * abs(y1 - y2) * 111) / height,
    )  # km / px
    zoom = _km_per_px_vs_zoom(max(max_bound, 4.777 / 1000))
    center = {"lon": (x1 + x2) / 2, "lat": (y1 + y2) / 2}
    fig.layout.mapbox.update(zoom=zoom, center=center)
    return fig


def _load_api_key():
    config_path = pathlib.Path.home() / ".mapbox" / "config"
    try:
        with open(config_path) as f:
            return f.read().rstrip()
    except FileNotFoundError:
        raise RuntimeError(f"mapbox access token should be placed as {config_path}")


def _zoom_vs_km_per_px(zoom):
    """
    Latitude ±40 (Cincinnati; Melbourne)
    https://docs.mapbox.com/help/glossary/zoom-level/#zoom-levels-and-geographical-distance
    """
    return 59959.436 * np.power(29979.718 / 59959.436, zoom) / 1000


def _km_per_px_vs_zoom(km_per_px):
    km_per_px *= 1000
    return (np.log(km_per_px / 59959.436)) / np.log(29979.718 / 59959.436)
