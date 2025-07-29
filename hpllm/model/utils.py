import jax
from jax import tree_util
import jax.numpy as jnp
from jax.experimental.shard import auto_axes
from jax.sharding import PartitionSpec as P

import dataclasses
from typing import TypeAlias , Any
import json
import os
from pathlib import Path

from hpllm.model.config import Model_Config
from hpllm.model.nn import TensorInfo , QuantTensor


def pytree_struct(cls, meta_fields: tuple = ()):
    """
    register_dataclass wrapper that automatically infers the data_fields

    Args:
        meta_fields (tuple, optional): meta_fields about the class , field which are supposed to be meta information. Defaults to ().

    """
    assert not dataclasses.is_dataclass(cls)

    cls = dataclasses.dataclass(cls)
    all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)

    return tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )


def hf_to_jax_config(
        hf_config : Any | dict[str , Any]
) -> Model_Config:

    _get = lambda x , k , default = None : (
        getattr(x , k , default) if not isinstance(hf_config , dict) else hf_config.get(k , default)
    )

    return Model_Config(
        embed=_get(hf_config, "hidden_size"),
        mlp_ffw_size=_get(hf_config, "intermediate_size", -1),
        moe_ffw_size=_get(hf_config, "moe_intermediate_size", -1),
        mlp_layer_idxs=_get(hf_config, "mlp_only_layers", []),
        q_heads=_get(hf_config, "num_attention_heads"),
        kv_heads=_get(hf_config, "num_key_value_heads"),
        num_layers=_get(hf_config, "num_hidden_layers"),
        head_dim=_get(hf_config, "head_dim"),
        vocab_size=_get(hf_config, "vocab_size"),
        norm_eps=_get(hf_config, "rms_norm_eps"),
        moe_experts_per_tok=_get(hf_config, "num_experts_per_tok"),
        moe_num_experts=_get(hf_config, "num_experts"),
        max_seq_len=128,
        dtype=jnp.bfloat16,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rope_theta=_get(hf_config, "rope_theta"),
    )


def load_tokenizer(
        tokenizer_path : str,
        tokenizer_config_path : str
):
    from transformers import PreTrainedTokenizerFast , AddedToken

    config = json.loads(Path(tokenizer_config_path).read_text())

    config = {
        k : AddedToken(**v) if isinstance(v , dict) and str(k).endswith("token") else v for (k ,v) in config.items()
    }

    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path) , **config)


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


def einsum(
    subscripts : str,
    lhs : jax.Array,
    rhs : jax.Array | QuantTensor,
    out_sharding : P | None = None,
):
    """
    Wrapper for jnp.einsum that can handle regular arrays and QuantTensor
    """
    #first condititon -> if array -> QuantTensor
    if is_type(rhs , QuantTensor):
        scale = jnp.expand_dims(rhs.scale , rhs.scale_expand_dims)

        if rhs.out_scaling:
            return(
                jnp.einsum(subscripts , lhs , rhs.quant , out_sharding=out_sharding) * scale
            )
        
        else:
            return jnp.einsum(subscripts , lhs * rhs , rhs.quant , out_sharding=out_sharding)
        

    #if normal array
    else:
        return jnp.einsum(subscripts , lhs , rhs , out_sharding=out_sharding)
    
