import jax
import jax.numpy as jnp
from jax.experimental.shard import auto_axes
from jax.sharding import PartitionSpec as P

from typing import Callable, Tuple, Dict
import dataclasses
from functools import partial

from hpllm.model.config import Model_Config
from hpllm.model.sharding import logical_to_sharding
from hpllm.model.utils import pytree_struct

# helper functions
is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (
    type(x).__module__ == cls.__module__
)

is_param = lambda x: is_type(x, TensorInfo)

count_left_padding = lambda ids, pad_id=0: auto_axes(
    lambda segment_ids: jnp.sum(
        jnp.cumsum(jnp.flip(segment_ids != 0, -1), axis=-1) > 0, -1
    ),
    out_sharding=P(None),
)(ids)


lenght_minus_padding = lambda segment_ids: auto_axes(
    lambda segment_ids: jnp.sum(
        jnp.cumsum(jnp.flip(segment_ids != 0, -1), axis=-1) > 0, -1
    ),
    out_sharding=P(None),
)(segment_ids)

which_platform = lambda cfg: cfg.mesh.devices.reshape(-1)[0].platform


@partial(pytree_struct , meta_fields=("shape" , "logical_axes" , "initializer" , "metadata"))
class TensorInfo:
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
class QuantTensor:
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

    quant: jax.Array | TensorInfo
    scale: jax.Array | TensorInfo
    out_scaling: bool = False
    scale_expand_dims: int | tuple[int, ...] = ()
    shape = property(lambda self: self.quant.shape)
    ndim = property(lambda self: self.quant.ndim)


def einsum(
    subscripts: str,
    lhs: jax.Array,
    rhs: jax.Array | QuantTensor,
    out_sharding: P | None = None,
):
    """
    Wrapper for jnp.einsum that can handle regular arrays and QuantTensor
    """
    # first condititon -> if array -> QuantTensor
    if is_type(rhs, QuantTensor):
        scale = jnp.expand_dims(rhs.scale, rhs.scale_expand_dims)

        if rhs.out_scaling:
            return (
                jnp.einsum(subscripts, lhs, rhs.quant, out_sharding=out_sharding)
                * scale
            )

        else:
            return jnp.einsum(
                subscripts, lhs * rhs, rhs.quant, out_sharding=out_sharding
            )

    # if normal array
    else:
        return jnp.einsum(subscripts, lhs, rhs, out_sharding=out_sharding)


class Module:
    """
    Base class for initializing primitives
    Main purpose is to define component information , sharding information and initialize weight

    Raises:
        NotImplementedError: if abstract method is not defined

    """

    @classmethod
    def abstract(cls, cfg: Model_Config, *args, **kw):
        """
        Define the component of each primitive

        Args:
            cfg (Model_Config): instance of Model Config

        Return : 
            PyTree object of cls

        """
        
        raise NotImplementedError

    @classmethod
    def shardings(cls, cfg: Model_Config, *args, **kw):
        """
        shard the elements along the specified dimension
        """
        assert type(cfg) == Model_Config
        
        abstract = cls.abstract(cfg, *args, **kw)

        return jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

    @classmethod
    def init(cls, key: jax.random.PRNGKey, cfg: Model_Config, *args, **kw):
        """
        Initialize each parameter defined in abstract

        Args:
            key (jax.random.PRNGKey)
            cfg (Model_Config)

        Returns:
            PyTree object containing the whole weights
        """
        #get the abstract instance of the class
        abstract = cls.abstract(cfg, *args, **kw)

        #get the sharding 
        shardings = jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )


        #main init function -> initialize the parameter 
        @partial(jax.jit, out_shardings=shardings)
        def _init():
            """
            Initialize the parameter of each leave of the pytree a

            Returns:
                _type_: _description_
            """
            # find total number of leaves
            num_leaves = len(jax.tree.leaves(abstract, is_leaf=is_param))
            # unqiue key for each leaf
            key_iter = iter(jax.random.split(key, num_leaves))

            # take in TensorInfo object and initialize each of its parameters
            return jax.tree.map(
                lambda info: info.initializer(next(key_iter), info.shape, info.dtype),
                abstract,
                is_leaf=is_param,
            )

        return _init()
