import numpy as np
import plotly

import plotly_utility.express as pux

if __name__ == "__main__":
    np.random.seed(0)

    y = np.arange(10)
    x = np.array(
        [
            np.random.normal(loc=np.random.normal(i, scale=0.5), scale=1, size=1000)
            for i in y
        ]
    )

    pux.ridgeline_plot(x, y).show()
