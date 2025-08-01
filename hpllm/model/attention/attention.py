import jax
import jax.numpy as jnp

from hpllm.model.config import Model_Config
from hpllm.model.utils import pytree_struct
from hpllm.model.nn import Module , TensorInfo , QuantTensor
from hpllm.model.sharding import logical_to_sharding
from hpllm.model.ffn.linear import MLPLayer
from hpllm.model.ffn.moe import MOELayer


@pytree_struct
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
                (cfg.embed_dim , cfg.n_heads ,cfg.head_dim),
                cfg.dtype,
                ("qkv_embed" , "n_heads" , "head_dim"), # based on ShardingRule
                _init(1,2)
            ),
            Wk = TensorInfo(
                (cfg.embed_dim , cfg.n_heads , cfg.head_dim),
                cfg.dtype,
                ("qkv_embed" , "n_heads" , "head_dim"),
                _init(1,2),
            ),
            Wv = TensorInfo(
                (cfg.embed_dim , cfg.n_heads , cfg.head_dim),
                cfg.dtype,
                ("qkv_embed" , "n_heads", "head_dim"),
                _init(1,2)
            ),
            Wo = TensorInfo(
                (cfg.n_heads , cfg.head_dim , cfg.embed_dim),
                cfg.dtype,
                ("qkv_embed" , "qkv_embed"),
                _init(1),

            )

        )

        return layer


        
@pytree_struct
class TransformerLayer(Module):

    ffw : MOELayer | MLPLayer # forward layer
    attn : AttentionLayer
    norm_pre_gamma : jax.Array | TensorInfo
    norm_post_gamma : jax.Array | TensorInfo

    @classmethod
    def abstract(cls , cfg : Model_Config , layer_index : int):
        #check if a layer is moe or not
        is_moe = cfg.moe_ffw_size > 0 and (layer_index not in cfg.mlp_layer_idxs)

        layer = TransformerLayer(
            ffw = MOELayer.abstract(cfg) if is_moe else MLPLayer.abstract(cfg),
            attn = AttentionLayer.abstract(cfg),
            norm_pre_gamma = TensorInfo(
                (cfg.embed_dim),
                cfg.dtype,
                ("act_embed",),
                jax.nn.initializers.constant(1.0),
            ),
            norm_post_gamma = TensorInfo(
                (cfg.embed_dim),
                cfg.dtype,
                ("act_embed",),
                jax.nn.initializers.constant(1.0),
            ),
        )

        return layer
    
