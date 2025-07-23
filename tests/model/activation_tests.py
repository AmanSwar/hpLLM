import torch
import jax
from hpllm.model import activation

from typing import List

def swish_test(
        x : jax.Array

):
    
    swish_out_jax = jax.nn.swish(x)
    swish_out_custom = activation.swish(x)

    return (swish_out_custom == swish_out_jax)


if __name__ == "__main__":
    key = jax.random.key(69)
    test_arr = jax.random.normal(key , shape=(1,10000))

    ans = swish_test(test_arr)
    print(ans)
    