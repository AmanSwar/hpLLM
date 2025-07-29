import jax
import jax.numpy as jnp
from jax.experimental.shard import auto_axes
from jax.sharding import PartitionSpec as P

from typing import Callable, Tuple, Dict
import dataclasses
from functools import partial

from config import Model_Config
from sharding import logical_to_sharding
from utils import is_param, pytree_struct


class ArrayInfo:
    """
    Dataclass for all jax.Arrays containing fundemental info
    Attributes:
        shape: shape of the array
        dtype: data type of elements in the array
        logical_axes: logical axes / dimension names in the array
        initializer: jax.nn.initializer function to initialize the data of array
        metadata: metadata about array
    """

    shape: tuple[int, ...]
    dtype: "jnp.dtype"
    logical_axes: tuple
    initializer: Callable | None = None
    metadata: dict = dataclasses.field(default_factory=dict)


@partial(pytree_struct, meta_fields=("out_scaling", "scale_expand_dims"))
class QuantArray:
    """
    Dataclass for quantization of model parameters
    Attributes:
        quant = jax.Array which holds the quantized object
        scale = holds the scaling factor to dequantize quant array (shape = shape(quantized array) - (minus) the shape along with the array is quantized)
        out_scaling = a flag which controls whether scale factor is applied to the output of a computation or input (for numerical stability during computation)
        scale_expand_dims = specifies dimension along with scale arrayu should be expanded to match the shape of quant array
        shape = python property erturning the shape of quant array
        ndim = python property returing number of dimensions of quant array
    """

    quant: jax.Array | ArrayInfo
    scale: jax.Array | ArrayInfo
    out_scaling: bool = False
    scale_expand_dims: int | tuple[int, ...] = ()
    shape = property(lambda self: self.quant.shape)
    ndim = property(lambda self: self.quant.ndim)


class Init:

    @classmethod
    def abstract(cls, cfg: Model_Config, *args, **kw):
        raise NotImplementedError

    @classmethod
    def shardings(cls, cfg: Model_Config, *args, **kw):
        abstract = cls.abstract(cfg, *args, **kw)

        return jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

    @classmethod
    def init(cls, key: jax.random.PRNGKey, cfg: Model_Config, *args, **kw):
        abstract = cls.abstract(cfg, *args, **kw)
        shardings = jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

        @partial(jax.jit, out_shardings=shardings)
        def _init():
            # find total number of leaves
            num_leaves = len(jax.tree.leaves(abstract, is_leaf=is_param))
            # unqiue key for each leaf
            key_iter = iter(jax.random.split(key, num_leaves))

            # take in ArrayInfo object and initialize each of its parameters
            return jax.tree.map(
                lambda info: info.initializer(next(key_iter), info.shape, info.dtype),
                abstract,
                is_leaf=is_param,
            )

        return _init()
