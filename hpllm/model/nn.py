import jax
import jax.numpy as jnp
from jax.experimental.shard import auto_axes
from jax.sharding import PartitionSpec as P

from typing import Callable , Tuple , Dict
import dataclasses
from functools import partial

from config import Model_Config
from sharding import logical_to_sharding
from utils import is_param

class ArrayInfo:
    shape : tuple[int , ...]
    dtype : "jnp.dtype"
    logical_axes : tuple
    initializer : Callable | None = None
    metadata : dict = dataclasses.field(default_factory=dict)


class Init:

    @classmethod
    def abstract(cls , cfg : Model_Config , *args , **kw):
        raise NotImplementedError
    
    
    @classmethod
    def shardings(
        cls , 
        cfg : Model_Config,
        *args,
        **kw
    ):
        abstract = cls.abstract(cfg , *args , **kw)

        return jax.tree.map(
            lambda info : logical_to_sharding(info.logical_axes , cfg.mesh , cfg.rules),
            abstract,
            is_leaf=is_param
        )
    

    @classmethod
    def init(
        cls,
        key : jax.random.PRNGKey,
        cfg : Model_Config,
        *args,
        **kw
    ):
        abstract = cls.abstract(cfg , *args , **kw)
        shardings = jax.tree.map(
            lambda info : logical_to_sharding(info.logical_axes , cfg.mesh , cfg.rules),
            abstract,
            is_leaf=is_param
        )

        @partial(jax.jit , out_shardings=shardings)
        def _init():
            # find total number of leaves
            num_leaves = len(
                jax.tree.leaves(abstract , is_leaf=is_param)
            ) 
            #unqiue key for each leaf
            key_iter = iter(jax.random.split(key , num_leaves))

            #take in ArrayInfo object and initialize each of its parameters
            return jax.tree.map(   
                lambda info : info.initializer(next(key_iter) , info.shape , info.dtype),
                abstract,
                is_leaf=is_param
            )
        
        return _init()
    
