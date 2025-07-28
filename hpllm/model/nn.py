import jax
import jax.numpy as jnp

from typing import Callable , Tuple , Dict
import dataclasses
from jax.experimental.shard import auto_axes
from jax.sharding import PartitionSpec as P


class ArrayInfo:
    shape : tuple[int , ...]
    dtype : "jnp.dtype"
    logical_axes : tuple
    initializer : Callable | None = None
    metadata : dict = dataclasses.field(default_factory=dict)



