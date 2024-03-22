from dataclasses import dataclass, field
from functools import reduce, wraps
from typing import Callable, Iterable, Sequence, Union

import numpy as np
from numpy.linalg import inv as np_inv
from numpy.typing import NDArray

from models.base_model import BaseModel
from models.types import T, Params, Samples


@dataclass
class LinearMultivariateModel(BaseModel):
    """A linear multivariate model.
    
    Attributes:
        __params: [private] A sequence or ndarray that maps to the factors
            of model, elements starting from 0th index correspond to
            factors of input seq/ndarray, while the last element is the
            biasing term.
    
    """
    __params: Union[Params, int] = field(default_factory=lambda: 2)
    
    def __post_init__(self):
        if isinstance(self.__params, Sequence):
            self.__params = np.array(self.__params)
        elif isinstance(self.__params, int):
            self.__params = np.zeros(self.__params)

        assert self.__params.ndim == 1, \
            f"Params in {self.__class__} should be a 1-d array/vector"
    
    @staticmethod
    def validate_input(fn):
        """Decorator used for input seq validation.
        
        The input seq's or array's length should be one less than the parameter
        seq's or array's length, since the last element in __params corresponds to
        the bias.
        
        It also handles promotion of scalar generics T to sequence to provide
        a consistent interface on passing input sequences/iterables.
        
        Args:
            fn: The funciton being decorated, in most cases magic method call.
        """
        
        np_append_ones: Callable[[NDArray[T]], NDArray[T]] = lambda x: \
            np.append(x, np.ones((x.shape[0], 1), dtype=x.dtype), 1)
        
        @wraps(fn)
        def wrapper(self, in_arr, *args):
            if not isinstance(in_arr, (Sequence, Iterable, np.ndarray)):
                in_arr = np.array([in_arr])
            elif not isinstance(in_arr, np.ndarray):
                in_arr = np.array(in_arr)
            
            # promote from vector to ndarray
            if in_arr.ndim == 1:
                in_arr = in_arr.reshape(1, -1)

            assert in_arr.shape[1] == self.__params.shape[0] - 1, \
                "Mismatch between input and param length"

            # append in_vec with a column of ones to match params
            in_arr = np_append_ones(in_arr)

            return fn(self, in_arr, *args)
        return wrapper
    
    @property
    def params(self) -> NDArray[T]:
        return self.__params
    
    @validate_input
    def __call__(
        self,
        in_arr: Union[Sequence[T], Iterable[T], T, Samples]
    ) -> NDArray[T]:
        """ Perform evaluation of model on input 
        Args:
            in_arr: scalar, numpy array, or sequence input.
                It's dimensions must be N x k, where k is one less the length of model
                params
        """
        return in_arr @ self.__params
    
    @validate_input
    def update(
        self,
        in_arr: Union[Sequence[T], Iterable[T], T, Samples],
        out_arr: Union[Sequence[T], Iterable[T], T, Samples],
    ):
        """ Update params using matrix least squares method
            
            Formulation:
                A * x = b , where b are the targets vector
            and A are the input data points array.
            
            Least Squares:
                A.T * A * x = A.T *  b
                x_hat = (A.T * A)^-1 * A.T * b , where x_hat are
            the estimated model params
            
        
        Args:
            in_arr: scalar, numpy array, or sequence input (A).
                It's dimensions must be N x k, where k is one less the length of model
                params    
            out_arr: scalar, numpy array, or sequence output (b).
                It's dimensions must be N x 1
        """
        self.__params = np_inv((in_t := in_arr.T) @ in_arr) @ in_t @ out_arr

    def loss(
        self,
        in_arr: Union[Sequence[T], Iterable[T], T, Samples],
        out_arr: Union[Sequence[T], Iterable[T], T, Samples],
    ):
        """ Calulates L2 loss for each sample (row wise).
        
        Args:
            in_arr: scalar, numpy array, or sequence input.
                It's dimensions must be N x k, where k is one less the length of model
                params    
            out_arr: scalar, numpy array, or sequence output.
                It's dimensions must be N x 1
        """
        return ((d := self(in_arr) - out_arr) * d).flatten()  # L2 loss

    def total_loss(
        self,
        in_arr: Union[Sequence[T], Iterable[T], T, Samples],
        out_arr: Union[Sequence[T], Iterable[T], T, Samples],
    ):
        """ Calulates L2 total loss (scalar).

        Args:
            in_arr: scalar, numpy array, or sequence input.
                It's dimensions must be N x k, where k is one less the length of model
                params    
            out_arr: scalar, numpy array, or sequence output.
                It's dimensions must be N x 1
        """
        return ((d := self(in_arr) - out_arr).T @ d).flatten()  # total L2 loss 
        
