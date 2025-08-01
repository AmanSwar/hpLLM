import jax


from hpllm.model.attention.attention import AttentionLayer , TransformerLayer
from hpllm.model.ffn.linear import MLPLayer
from hpllm.model.ffn.moe import MOELayer

from tests.model.util import model_config

key = jax.random.PRNGKey(0)

def attention_init_test():

    attn_weights = AttentionLayer.init(key=key , cfg=model_config)

    print(attn_weights)
    print(type(attn_weights))


def mlp_init_test():

    mlp_weights = MLPLayer.init(key=key, cfg=model_config)

    print(mlp_weights)
    print(type(mlp_weights))


def moe_init_test():

    moe_weights = MOELayer.init(key=key, cfg=model_config)

    print(moe_weights)
    print(type(moe_weights))


def transformerlayer_init_test():

    transformer_layer_weights = TransformerLayer.init(key , cfg=model_config , layer_index=2)

    print(transformer_layer_weights)
    print(type(transformer_layer_weights))



if __name__ == "__main__":
    # moe_init_test()
    # mlp_init_test()
    transformerlayer_init_test()
