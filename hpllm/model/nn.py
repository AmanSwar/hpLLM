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


#helper functions
is_type = lambda x , cls : (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)

is_param = lambda x : is_type(x , ArrayInfo)

count_left_padding = lambda ids , pad_id = 0 : auto_axes(
    lambda segment_ids : jnp.sum(
        jnp.cumsum(jnp.flip(segment_ids != 0 , -1) , axis=-1) > 0 , 
        -1
    ),
    out_sharding=P(None),

)(ids)


lenght_minus_padding = lambda segment_ids : auto_axes (
    lambda segment_ids : jnp.sum(
        jnp.cumsum(jnp.flip(segment_ids != 0 , -1) , axis=-1) > 0 , -1
    ),
    out_sharding=P(None),

)(segment_ids)

which_platform = lambda cfg : cfg.mesh.devices.reshape(-1)[0].platform
