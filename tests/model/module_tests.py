import jax


from hpllm.model.attention.attention import AttentionLayer
from hpllm.model.ffn.linear import MLPLayer
from tests.model.util import model_config

key = jax.random.PRNGKey(0)
def attention_init_test():
    

    attn_weights = AttentionLayer.init(key=key , cfg=model_config)

    print(attn_weights)
    print(type(attn_weights))
    

def mlp_init_test():

    mlp_weights = MLPLayer.init(key=key , cfg=model_config)

    print(mlp_weights)
    print(type(mlp_weights))


if __name__ == "__main__":
    mlp_init_test()