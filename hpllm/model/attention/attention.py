import jax
import jax.numpy as jnp

from hpllm.model.config import Model_Config
from hpllm.model.utils import pytree_struct
from hpllm.model.nn import Module , TensorInfo , QuantTensor
from hpllm.model.sharding import logical_to_sharding

class AttentionLayer(Module):

    """
    (dataclas)Attention Module wrapper
    Fields:
        Wq : Weight Matrix for Query
        Wk : Weight Matrix for Key
        Wv : Weight Matrix for Value
        Wo : Weight Matrix for O
    
    """

    Wq : jax.Array | TensorInfo | QuantTensor
    Wk : jax.Array | TensorInfo | QuantTensor
    Wv : jax.Array | TensorInfo | QuantTensor
    Wo : jax.Array | TensorInfo | QuantTensor


    @classmethod
    def abstract(cls, cfg: Model_Config):
        _init = lambda *out_axes : jax.nn.initializers.he_normal(
            in_axis=0,
            out_axis=out_axes
        )

        layer = AttentionLayer(
            Wq = TensorInfo(
                (cfg.embed , cfg.q_heads ,cfg.head_dim),
                cfg.dtype,
                ("qkv_embed" , "kv_head" , "head_dim"), # based on ShardingRule
                _init(1,2)
            ),
            Wk = TensorInfo(
                (cfg.embed , cfg.kv_heads , cfg.head_dim),
                cfg.dtype,
                ("qkv_embed" , "kv_heads" , "head_dim"),
                _init(1,2),
            ),
            Wv = TensorInfo(
                (cfg.embed , cfg.kv_heads , cfg.head_dim),
                cfg.dtype,
                ("qkv_embed" , "kv_heads", "head_dim"),
                _init(1,2)
            ),
            Wo = TensorInfo(
                (cfg.q_heads , cfg.head_dim , cfg.embed),
                cfg.dtype,
                ("qkv_embed" , "qkv_embed"),
                _init(1),

            )

            return layer
        )


        