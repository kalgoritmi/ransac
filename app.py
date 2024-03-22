#!/bin/env python3

from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate

from models.linear_multivariate import LinearMultivariateModel
from utils import gen_regression_data, history_figure, regression_figure
from ransac.regressor import RansacRegressor


def regression():
    params = [2, 1]
    lin_model = LinearMultivariateModel(params)

    data_x, data_y = gen_regression_data(lin_model)

    regressor = RansacRegressor(lin_model, relative_tolerance=1e-3)
    regressor(data_x, data_y)  # fit regressor 

    print(f"\nBest model params: {regressor.model.params.flatten()},"
          f" iteration: #{regressor.best_idx}, loss: {regressor.best_loss}\n")

    return data_x, data_y, regressor


app = Dash()

data_x, data_y, regressor = regression()

app.layout = html.Div([
    dcc.Graph(
        figure=regression_figure(
            data_x,
            data_y,
            regressor.model,
        )
    ),
    dcc.Interval(
        id="plot-interval",
        interval=100,
        n_intervals=0
    ),
    html.Div(
        id="iteration",
        style={"text-align": "center", "font-family": "sans-serif"}
    ),
    dcc.Graph(
        id="plot",
    ),
])

history_figure_closure = history_figure(data_x, data_y, regressor)


@app.callback(
    Output("plot", "figure"),
    Input("plot-interval", "n_intervals")
)
def update_plot(n_interval):
    try:
        return history_figure_closure(regressor.history[n_interval].model, n_interval)
    except IndexError:
        raise PreventUpdate 


@app.callback(
    Output("iteration", "children"),
    Input("plot-interval", "n_intervals")
)
def update_iteration(n_interval):
    try:
        return html.Span(
            f"Best Candidate @ Iteration #{regressor.history[n_interval].iteration}",
            style={
                "color": "#55CC80"
                if regressor.history[n_interval].best_flag
                else "salmon",
            },
        )
    except IndexError:
        return html.Span(
            f"Best model found in iteration #{regressor.best_idx}",
            style={
                "color": "#55CC80",
            },
        )


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
    