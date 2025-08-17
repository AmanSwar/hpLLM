import jax
import jax.numpy as jnp

@jax.jit
def rmsnorm(x : jax.Array , gamma : jax.Array , eps=1e-6) -> jax.Array:
    """
    Applied RMS Normalization

    Args:
        x (jax.Array) : The input array to which RMS norm is applied
        gamma (jax.Array): Learnable parameter
        eps (_type_, optional): small value to avoid division by 0, Defaults to 1e-6.
    Returns: normalized x
    """    

    rms = jnp.sqrt(jnp.mean(jnp.square(x) , axis=-1 , keepdims=True) + eps)
    normalized_x  = x / rms
    return gamma * normalized_x


