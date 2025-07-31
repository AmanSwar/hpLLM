import jax
import jax.numpy as jnp

from typing import Tuple , Callable 

from hpllm.model.config import Model_Config
from hpllm.model.nn import Module , TensorInfo
from hpllm.model.utils import pytree_struct



class MLPLayer(Module):
    """
    MLP layer wrapper 
    Fields:
        Wgate -> Weight matrix of gating component used in Gate based activation functions
        Wup and Wdown -> Standard weight matrices for 2 layer MLP as done in original transformer architecture
    """    

    Wgate : jax.Array | TensorInfo
    Wup : jax.Array | TensorInfo
    Wdown : jax.Array | TensorInfo

    
    @classmethod
    def abstract(cls, cfg: Model_Config):
        _init = lambda *out_axes : jax.nn.initializers.he_normal(
            in_axis=0,
            out_axis=out_axes
        )


        layer = MLPLayer(
            Wgate = TensorInfo(
                (cfg.embed_dim , cfg.mlp_ffw_size),
                cfg.dtype,
                ("mlp_up_embed" , "mlp_up_ffw"),
                _init(1)
            )

            Wup = TensorInfo(
                (cfg.embed_dim, cfg.mlp_ffw_size),
                cfg.dtype,
                ("mlp_up_embed" , "mlp_up_ffw"),
                _init(1)
            )

            Wdown = TensorInfo(
                (cfg.mlp_ffw_size , cfg.embed_dim),
                cfg.dtype,
                ("mlp_down_ffw" , "mlp_down_embed"),
                _init(1)
            )

        )

        return layer

        


