import jax


from hpllm.model.config import Model_Config
from hpllm.model.nn import Module , TensorInfo
from hpllm.model.utils import pytree_struct

@pytree_struct
class MOELayer(Module):

    Wrouter : jax.Array | TensorInfo #router weights
    Wgate : jax.Array | TensorInfo # gating layer weights
    Wup : jax.Array | TensorInfo # upward projection weights
    Wdown : jax.Array | TensorInfo #downward projection weight

    @classmethod
    def abstract(cls, cfg: Model_Config):
        # expert initializers
        _expert_init = jax.nn.initializers.he_normal(in_axis=0 , out_axis=(1,2))

        # router init
        _h_init = jax.nn.initializers.he_normal(in_axis=0 , out_axis=1)

        dtype = cfg.dtype

        layer = MOELayer(
            Wrouter=TensorInfo(
                (cfg.embed_dim, cfg.moe_num_experts),
                cfg.moe_gate_dtype,
                ("moe_e_up_embed", None),
                _h_init,
            ),
            Wgate=TensorInfo(
                (cfg.moe_num_experts, cfg.embed_dim, cfg.moe_ffw_size),
                cfg.dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _expert_init
            ),
            Wup = TensorInfo(
                (cfg.moe_num_experts , cfg.embed_dim , cfg.moe_ffw_size),
                cfg.dtype,
                ("moe_e_experts" , "moe_e_up_embed" , "moe_e_up_ffw"),
                _expert_init
            ),
            Wdown=TensorInfo(
                (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed_dim),
                cfg.dtype,
                ("moe_e_experts", "moe_e_up_ffw", "moe_e_down_embed"),
                _expert_init
            ),
        )

        return layer
