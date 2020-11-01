import numpy as np
import numpy_utility as npu
import collections.abc
import plotly
import os
import webbrowser
import re
import functools


__all__ = ["figures_to_html", "plot"]


DEFAULT_CHART_CONFIG = {
        'modeBarButtons': [
            [
                'toImage',
                'sendDataToCloud',
                "select2d",
                'zoom2d',
                'pan2d',
                'zoomIn2d',
                'zoomOut2d',
                'autoScale2d',
                'resetScale2d',
                'toggleSpikelines',
                'hoverClosestCartesian',
                'hoverCompareCartesian'
            ]
        ],
        "showSendToCloud": True,
        "plotlyServerURL": "https://chart-studio.plotly.com",
        "toImageButtonOptions": dict(
            format="svg"
        ),
        "editable": True
    }


depth = lambda L: isinstance(L, (list, np.ndarray)) and max(map(depth, L)) + 1


def _get_n_rows_cols(figs):
    if depth(figs) == 2:
        n_cols = max(len(row_figs) if row_figs is not None else 0 for row_figs in figs)
        n_rows = len(figs)
    else:
        n_cols = len(figs)
        n_rows = 1
    return n_rows, n_cols


def _write_figs_to(dashboard, figs):
    n_rows, n_cols = _get_n_rows_cols(figs)

    if depth(figs) == 2:
        figs = [fig if fig is not None else {} for row_figs in figs for fig in row_figs]

    # n_rows = (len(figs) - 1) // n_cols + 1
    if n_rows == 1:
        p = 100
    elif n_rows == 2:
        p = 50
    elif n_rows == 3 or n_rows == 4:
        p = 25
    else:
        raise NotImplementedError

    dashboard.write(f'<div class="row no-gutters row-cols-{n_cols} h-{p}">')
    add_js = True
    for fig in figs:
        dashboard.write('<div class="col">')
        if isinstance(fig, collections.abc.Sequence):
            _write_figs_to(dashboard, fig)
        elif fig == {}:
            pass
        else:
            inner_html = plotly.offline.plot(
                fig, include_plotlyjs=add_js, output_type='div', config=DEFAULT_CHART_CONFIG
            )
            dashboard.write(f'{inner_html}')
        dashboard.write('</div>')
        add_js = False
    dashboard.write("</div>")


def figures_to_html(figs, filename="temp-plot.html", auto_open=True, editable=True):
    if isinstance(figs, plotly.graph_objs.Figure):
        figs = [figs]
    elif npu.is_array(figs):
        pass
    else:
        raise ValueError(type(figs))

    if filename is None or filename == "":
        filename = "temp-plot.html"

    dashboard = open(filename, 'w')
    head = '''    
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
    '''
    scripts = '''
    <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>    
    '''

    dashboard.write(f'''
    <html>
        <head>{head}</head>
        <body>
            <div class="container-fluid">
    ''')

    _write_figs_to(dashboard, figs)

    dashboard.write(f'''
            </div>
            {scripts}
        </body>
    </html>
    ''')

    if auto_open:
        url = "file://" + os.path.abspath(filename)
        webbrowser.open(url)


def plot(figs, filename="temp-plot.html", auto_open=True, editable=True,
         hovertext_format=".2f", replace_none_titles=True):
    if isinstance(figs, plotly.graph_objs.Figure):
        figs = [figs]
    elif npu.is_array(figs):
        pass
    else:
        raise ValueError(type(figs))

    for fig in np.ravel(figs):
        for trace in fig.data:
            if hovertext_format is not None and trace.hovertemplate is not None:
                vars = npu.ma.from_jagged_array([m.split(":") for m in re.findall("%{([^}]+)}", trace.hovertemplate)], 2)
                float_vars = vars[[npu.is_floating(np.array(eval(f"trace.{var}", {"trace": trace}))) for var, _ in vars]]
                trace.hovertemplate = functools.reduce(
                    lambda s, rpl: s.replace(*rpl),
                    zip(np.char.add(np.char.add("%{", npu.char.join(":", float_vars, axis=1)), "}"),
                        (f"%{{{fv}:{hovertext_format}}}" for fv, _ in float_vars)),
                    trace.hovertemplate
                )
            if replace_none_titles:
                if fig.layout.title.text is None:
                    fig.layout.title.text = ""
                for axis in (fig.layout[k] for k in fig.layout if k.startswith("xaxis") or k.startswith("yaxis")):
                    if axis.title.text is None:
                        axis.title.text = ""

    figures_to_html(figs, filename, auto_open, editable)