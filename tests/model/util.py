import jax
import jax.numpy as jnp
from jax.sharding import Mesh


import os

from hpllm.model.config import Model_Config
from hpllm.model.sharding import ShardingRules

# devices = jax.local_devices()

# if len(devices) < 2:
#     devices = jax.devices()


def sim_multiCPU_dev(device_count: int = 8):
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"

    os.environ["XLA_FLAGS"] = flags

    # disable CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


sim_multiCPU_dev(8)
devices = jax.devices()


mesh = jax.make_mesh((2,4) , axis_names=("x", "y"))


BATCH_AXIS_NAME = "x"
EXPERT_AXIS_NAME = "y"
TENSOR_ONLY_AXIS_NAME = "y"
ATTN_HEADS_AXIS_NAME = "y"
TENSOR_AXIS_NAME = ("y")

sharding_rule = ShardingRules(
    batch=BATCH_AXIS_NAME,
    sequence=None,  # sequence lenghtś
    act_embed=None,  # activation along the embedding dimension
    act_heads=None,  # activations just before mergeing all the heads
    head_dim=None,  # per head dimensionality
    # attention1
    qkv_embed=None,  #
    n_heads=ATTN_HEADS_AXIS_NAME,
    o_heads=ATTN_HEADS_AXIS_NAME,
    o_embed=None,
    # MLP layer
    mlp_up_embed=None,
    mlp_up_ffw=TENSOR_AXIS_NAME,
    mlp_down_ffw=TENSOR_AXIS_NAME,
    mlp_down_embed=None,
    # MoE layer
    moe_e_experts=EXPERT_AXIS_NAME,
    moe_e_up_embed=None,
    moe_e_up_ffw=TENSOR_ONLY_AXIS_NAME,
    moe_e_down_ffw=TENSOR_ONLY_AXIS_NAME,
    moe_e_down_embed=None,
    moe_e_tp=(TENSOR_ONLY_AXIS_NAME),  # moe forward function tensor parallelism
    moe_e_ep=EXPERT_AXIS_NAME,  # moe forward function expert parallelism
    # vocab
    vocab_in=None,
    vocab_out=TENSOR_AXIS_NAME,
)


model_config = Model_Config(
    dtype=jnp.bfloat16,
    embed_dim=128,
    n_heads=8,
    num_layers=2,
    head_dim=32,
    vocab_size=1024,
    max_seq_len=512,
    causal=True,
    mlp_ffw_size=256,
    mlp_layer_idxs=[0, 1],  
    moe_ffw_size=-1,
    moe_num_experts=0,
    moe_experts_per_tok=0,
    mesh=mesh,
    rules=sharding_rule,  # 
)
