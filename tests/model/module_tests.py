import jax


from hpllm.model.attention.attention import AttentionLayer
# from hpllm.model.ffn.linear import MLPLayer
from tests.model.util import model_config



def attention_init_test():
    key = jax.random.PRNGKey(0)

    attn_weights = AttentionLayer.init(key=key , cfg=model_config)

    print(attn_weights)
    print(type(attn_weights))
    



if __name__ == "__main__":
    attention_init_test()