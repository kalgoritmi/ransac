from typing import Any, Sequence, Tuple, TypeVar, Union

from numpy.typing import NDArray

T = TypeVar("T", bound=float | int)

Samples = Union[NDArray[T], Sequence[T]]

Params = Union[NDArray[T], Sequence[T]] 
