import jax
from jax import tree_util

import dataclasses
from typing import TypeAlias

AxisName: TypeAlias = str | tuple[str, ...] | None


@dataclasses.dataclass
class ShardingRules:
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
