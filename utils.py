from typing import Any

from numpy.random import default_rng
from numpy.random._generator import Generator as NpRNG
from numpy.typing import NDArray
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as pgo

from models.base_model import BaseModel
from models.types import T

RNG = default_rng(seed=10)  # for reproducibility

clamp = lambda x, lower, upper: min(max(x, lower), upper)

l2_normalize = lambda vec: vec / np.sqrt(vec.T @ vec)


def gen_regression_data(
    model: BaseModel,
    n_points: int = 100,
    ratio: float = .6,
    rng: NpRNG = RNG,
    mag: int = 10,
    var_noise: int = 2,
    noise: int = 10,
):
    assert 0 < ratio < 1, "Ratio should be in range (0, 1)"

    shape = (int(ratio * n_points), len(model) - 1)
    outlier_shape = (int((1 - ratio) * n_points), len(model) - 1)

    data_x = mag * (2 * rng.random(shape) - 1)

    data_y = model(data_x).reshape(-1, 1)
    
    """
    # add variational noise
    data_x += var_noise * (2 * rng.random(shape) - 1)
    
    data_y += var_noise * (2 * rng.random(shape) - 1)
    """
    
    # concatenate gaussian distributed outlier points 
    data_x = np.vstack((
        data_x,
        rng.normal(scale=noise, size=outlier_shape),
    ))
    
    data_y = np.vstack((
        data_y,
        rng.normal(scale=noise, size=outlier_shape),
    ))
    
    return data_x, data_y


def regression_figure(
    data_x: NDArray[T],
    data_y: NDArray[T],
    model: BaseModel = None,
):
    data = [
        pgo.Scatter(
            x=data_x.flatten().tolist(),
            y=data_y.flatten().tolist(),
            mode='markers',
            name='Data',
        ),
    ]
    
    if model is not None:
        data.append(
            pgo.Scatter(
                x=data_x.flatten().tolist(),
                y=model(data_x).flatten().tolist(),
                mode='lines+markers',
                name='Best Model',
            )
        )
    
    return pgo.Figure(
        data,
        layout_yaxis_range=[np.min(data_y) - 10, np.max(data_y) + 10],
        layout_title_text="Linear Univariate Model Regression using RANSAC"
    )


def history_figure(
    data_x: NDArray[T],
    data_y: NDArray[T],
    regressor: Any,
):
    losses, iterations = \
        [e.loss for e in regressor.history], [e.iteration for e in regressor.history]

    def closure(model: BaseModel, iteration: int) -> pgo.Figure:
        fig = make_subplots(rows=2, cols=1)
        
        fig.add_trace(
            pgo.Scatter(
                x=data_x.flatten().tolist(),
                y=data_y.flatten().tolist(),
                mode='markers',
                name='Data',
            ),
            row=1,
            col=1,
        )
        
        fig.add_trace(
            pgo.Scatter(
                x=data_x.flatten().tolist(),
                y=model(data_x).flatten().tolist(),
                mode='lines+markers',
                name='Best Candidate',
            ),
            row=1,
            col=1,
        )
    
        fig.add_trace(
            pgo.Scatter(
                x=iterations,
                y=losses,
                mode="lines",
                name="Best Candidate Losses",
            ),
            row=2,
            col=1,
        )
        
        fig.add_vline(
            x=iterations[iteration],
            line_dash="dash", line_color="salmon", row=2, col=1
        )
        
        fig.add_trace(
            pgo.Scatter(
                x=[iterations[iteration]],
                y=[losses[iteration]],
                mode="markers",
                name="Current Loss",
            ),
            row=2,
            col=1,
        )
    
        fig.update_layout(
            yaxis_range=[np.min(data_y) - 10, np.max(data_y) + 10],
            title_text="Best Candidate Replay with Loss"
        )
    
        return fig
    return closure
