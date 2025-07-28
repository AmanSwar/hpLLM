import jax

import dataclasses
from typing import TypeAlias


AxisName: TypeAlias = str | tuple[str, ...] | None
Axes : TypeAlias = tuple[AxisName , ...]

@dataclasses.dataclass
class ShardingRules:
    """
    Mapping from logical data axis to physical mesh axis
    """

    batch: AxisName
    sequence: AxisName

    # activation
    act_embed: AxisName
    act_heads: AxisName

    # attention
    head_dim: AxisName
    qkv_embed: AxisName
    q_heads: AxisName
    kv_heads: AxisName
    o_heads: AxisName
    o_embed: AxisName

    # MLP layer
    mlp_up_embed: AxisName
    mlp_up_ffw: AxisName
    mlp_down_ffw: AxisName
    mlp_down_embed: AxisName

    # MOE layer
    moe_e_experts: AxisName
    moe_e_up_embed: AxisName
    moe_e_up_ffw: AxisName
    moe_e_down_ffw: AxisName
    moe_e_down_embed: AxisName
    moe_e_tp: AxisName  # MOE forward function for Tensor Parallelism
    moe_e_ep: AxisName  # MOE forward func for expert parallism

    # vocab
    vocab_in: AxisName
    vocab_out: AxisName



