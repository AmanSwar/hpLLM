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


def logical_to_physcial(
        logical_axes : Axes,
        rules : ShardingRules
) -> jax.sharding.PartitionSpec:
    """
    Returns how to physically shard a given sequence of logical array

    Args:
        logical_axes (Axes) : logical dimensions 
        rules (ShardingRules) : Sharding rules defining which logical dimension maps to which axes on mesh

    Raises:
        ValueError: if all values in spec are not unqiue
        NotImplementedError : if any dimension of logical axes is missing in Sharding Rules

    Returns:
        jax.sharding.PartitionSpec 
    """    
    try:
        spec = [getattr(rules , axis) if axis is not None else None for axis in logical_axes]
    except NotImplementedError:
        raise NotImplementedError(f"Logical axes not found in Sharding rules  Logical axes : {logical_axes} , Sharding rules : {[field.name for field in dataclasses.fields(ShardingRules)]}")
    flat_axes = jax.tree.leaves(spec)

    if len(set(flat_axes)) != len(flat_axes):
        raise ValueError(
            f"Colliding physical axes from translating logical spec {logical_axes} -> {spec}"
        )
    
    return jax.sharding.PartitionSpec(*spec)


def logical_to_sharding(
        logical_axes : Axes,
        mesh : jax.sharding.Mesh,
        rules : ShardingRules
) -> jax.sharding.Sharding:
    """
    returns the sharding for a given sequence of logical array dim

    Args:
        logical_axes (Axes): logical dimensions
        mesh (jax.sharding.Mesh): Mesh object
        rules (ShardingRules): Sharding Rules

    Returns:
        jax.sharding.Sharding
    """

    assert mesh is not None
    return jax.sharding.NamedSharding(mesh , logical_to_physcial(logical_axes , rules))


