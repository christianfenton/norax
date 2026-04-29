import jax
import jax.numpy as jnp

from norax import FNO

key = jax.random.key(0)

model = FNO(key, channels_in=2, channels_out=1, n_modes=(8,), width=16, depth=2)

x = jnp.ones((4, 32, 2))
y = jax.vmap(model)(x)

assert y.shape == (4, 32, 1), f"unexpected output shape: {y.shape}"
print("smoke test passed")
