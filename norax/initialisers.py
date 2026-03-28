import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray


def complex_glorot(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    in_axis: int = 1,
    out_axis: int = 0,
    dtype: jnp.dtype = jnp.result_type(float),
) -> Complex[Array, "*shape"]:
    """Glorot-scaled complex weight initialisation.

    Args:
        key: PRNG key
        shape: Weight tensor shape
        in_axis: Axis corresponding to input channels
        out_axis: Axis corresponding to output channels
        dtype: Real dtype for the components (float32 or float64)

    Returns:
        Complex array of the given shape (complex64 or complex128)
    """
    complex_dtype = jnp.result_type(dtype, 1j)

    fan_in = shape[in_axis]
    fan_out = shape[out_axis]

    key_r, key_theta = jax.random.split(key)

    minval = jnp.finfo(dtype).tiny
    uni = jax.random.uniform(key_r, shape, dtype=dtype, minval=minval)
    r = jnp.sqrt((2.0 / (fan_in + fan_out)) * -jnp.log(uni))

    theta = jax.random.uniform(
        key_theta, shape, dtype=dtype, minval=0.0, maxval=2.0 * jnp.pi
    )

    return (r * jnp.exp(1j * theta)).astype(complex_dtype)
