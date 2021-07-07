import plotly.express as px
import plotly.graph_objs as go
import plotly.express._core
import numpy_utility as npu


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
    else:
        args = px._core.build_dataframe(args, constructor)

    return args
