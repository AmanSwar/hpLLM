import jax
import jax.numpy as jnp

from typing import Tuple
    
@jax.custom_vjp
def swish(x : jax.Array , beta : jax.Array | None = None) -> jax.Array:
    """
    Swish(x) = x * sigmoid(x)
    sigmoid(x) = 1 / 1 + e(-x)

    """
    if beta == None:
        beta = jnp.ones(x.size)

    z = beta * x
    return x * jax.nn.sigmoid(z)


def swish_frwd(x : jax.Array, beta : jax.Array | None = None):
    fx = swish(x , beta)
    #for backward pass -> input and output
    residual = (x , beta , fx)
    return fx , residual


def swish_bkwd(residual : Tuple , grad) -> Tuple:

    x , beta , fx  = residual
    #caluclate the backward pass 
    # grad_x = sigmoid(beta*x) * (diff of sigmoid) * beta
    #diff of sigmoid -> sig x (1 - sig)
    z = beta * x
    diff_sigmoid = (jax.nn.sigmoid(z) * (1 - jax.nn.sigmoid(z))) * beta
    grad_x = jax.nn.sigmoid(z) * diff_sigmoid

    #grad_b = x * Sigmoid(bx) * diff_sigmoid *x
    grad_b = x * jax.nn.sigmoid(z) * diff_sigmoid * x

    grad_x = grad * grad_x
    grad_b = grad * grad_b

    return (grad_x , grad_b)




swish.defvjp(swish_frwd , swish_bkwd)



    
    