import plotly.express as px
import plotly.graph_objs as go
import plotly.express._core
import numpy_utility as npu
import numpy as np


__all__ = ["build_dataframe"]


def build_dataframe(args, constructor):
    if constructor == go.Histogram:
        not_has_labels = args["labels"] is None

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

        if not_has_labels:
            # Prevent that labels is set automatically if marginal=="rug"
            args["labels"] = None
    elif constructor in (go.Scatter, go.Scattergl):
        args = px._core.build_dataframe(args, constructor)
        if args["category_orders"] is None or args["color"] not in args["category_orders"]:
            args["category_orders"][args["color"]] = np.unique(args["data_frame"][args["color"]])
    else:
        args = px._core.build_dataframe(args, constructor)

    return args
