from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

from numpy.random import choice as np_choice
from numpy.typing import NDArray

from models import BaseModel, Samples
from models.types import T
from utils import clamp, l2_normalize

Entry = namedtuple("Entry", "model loss best_flag iteration")


class RansacRegressor:
    """
    
    Attributes:
        model: an object whose class implements the BaseModel interface
        relative_tolerance: relative error tolerance for model fitness as
            a good candidate
        max_iterations: maximum number of iterations to perform
        inlier_threshold: how many inlier points (i.d. points with loss less than rel_tol)
            a model must have to be a good candidate
        history: a list of entries that include, best candidate model, loss, iteration,
            best candidate flag
        best_idx: index of best model
        best_loss: loss of best model
        best_model: best model
    """
    @dataclass
    class RandomSampler:
        """ Facilitates lazy random sampling
        
        Attributes:
            in_data: input array, dimensions N x k, where k is one less the model
                params length
            out_data: output array, dimensions N x 1
            n_samples: the number of smaples to return each time it yields
            sampler: a callable that performs random sampling in bulk, defaults to
                numpy.random.choice  
        """
        in_data: NDArray[T]
        out_data: NDArray[T]
        n_samples: int
        sampler: Callable[[NDArray[T]], NDArray[T]] = np_choice
        
        def __post_init__(self):
            assert self.in_data.shape[0] == self.out_data.shape[0], \
                "Number of points in input and output data must match," \
                f" got {self.in_data.shape} & {self.out_data.shape}}}"
        
        def __iter__(self):
            return self
        
        def __enter__(self):
            return iter(self)
        
        def __exit__(self, *args):
            pass
        
        def __next__(self):
            sampled_idx = self.sampler(
                self.in_data.shape[0],
                self.n_samples,
                replace=False,
            )  # sample without replacement
            
            return self.in_data[sampled_idx, :], self.out_data[sampled_idx, :]
    
    def __init__(
        self,
        model: BaseModel,
        relative_tolerance: float = 1e-2,
        max_iterations: int = 1_000,
        inlier_threshold: Optional[int] = None,
    ):
        self.model = model
        self.rel_tol = relative_tolerance
        self.max_iter = max_iterations
        self.inlier_threshold = 5 * self.model.params.shape[0] \
            if inlier_threshold is None else inlier_threshold

        self.best_model, self.best_loss = None, None  # best model and best loss
        self.best_idx: int = []  # best model index
        self.history: List[Entry] = []  # historical best candidates for visualiztion 

    def __call__(self, in_data: NDArray[T], out_data: NDArray[T]):
        k = clamp(
            10 * self.model.params.shape[0],
            0,
            in_data.shape[0] // 3,
        )  # number of samples to be used in each iteration, ~ 10 times the params len

        with self.RandomSampler(in_data, out_data, k) as sampler:
            for i in range(self.max_iter):
                in_sampled, out_sampled = next(sampler)  # sample from data pool
                self.model.update(in_sampled, out_sampled)  # update model params
                
                # normalize loss, to facilitate relative comparison
                rel_loss = l2_normalize(self.model.loss(in_data, out_data))
                
                inlier_idx = rel_loss <= self.rel_tol  # threshold inlier indices
                in_inliers, out_inliers = in_data[inlier_idx], out_data[inlier_idx]
                
                if in_inliers.shape[0] > self.inlier_threshold:
                    # update model params using only inliers
                    self.model.update(in_inliers, out_inliers)
                    
                    # total loss
                    total_loss = self.model.total_loss(in_inliers, out_inliers)
                    
                    best_flag = False
                    if self.best_loss is None or total_loss < self.best_loss:
                        self.best_model = deepcopy(self.model)
                        self.best_loss = float(total_loss[0])
                        self.best_idx = i
                        best_flag = not best_flag
                
                    print(f"Iteration #{i} | Params: {self.model.params.flatten()}"
                          f" | Samples #{len(in_sampled)} | Total Loss: {total_loss}")
                
                    self.history.append(
                        Entry(deepcopy(self.model), total_loss[0], best_flag, i)
                    )
        
        # restore best model 
        self.model: BaseModel = deepcopy(self.best_model)
