import plotly.express as px
import plotly.graph_objs as go
import plotly.express._core
import numpy_utility as npu
import numpy as np


__all__ = ["build_dataframe"]


def build_dataframe(args: dict, constructor):
    if np.ma.isMaskedArray(args["data_frame"]):
        a = args.pop("data_frame")
        args["data_frame"] = a.data
        args = build_dataframe(args, constructor)
        args["data_frame"] = args["data_frame"][~npu.any(a.mask, axis="column")]
        return args

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
        if args["category_orders"] is None:
            args["category_orders"] = dict()
        if args["color"] is not None and args["color"] not in args["category_orders"]:
            color, indices = np.unique(args["data_frame"][args["color"]], return_index=True)
            args["category_orders"][args["color"]] = color[np.argsort(indices)]
    else:
        args = px._core.build_dataframe(args, constructor)

    return args
