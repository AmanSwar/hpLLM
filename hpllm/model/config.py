import jax
from jax import tree_util
import jax.numpy as jnp
from jax.sharding import Mesh

import dataclasses

from hpllm.model.sharding import ShardingRules
static_class = lambda cls :  tree_util.register_static(dataclasses.dataclass(cls))


@static_class
class Model_Config:

    dtype : "jnp.dtype"

    # basic config
    embed_dim: int  
    n_heads: int
    # kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int
    max_seq_len: int

    # type of attention
    causal: bool

    # MOE
    moe_ffw_size: int
    moe_experts_per_tok: int
    moe_num_experts: int
    moe_gate_dtype: "jnp.dtype" = jnp.float32
    ep_strategy: str = "decode"

    #MLP config
    mlp_ffw_size: int = -1
    mlp_layer_idxs: list[int] = dataclasses.field(default_factory=list)

    #distributed
    mesh : Mesh | None = None
    rules : ShardingRules = dataclasses.field(default_factory=ShardingRules)
    
