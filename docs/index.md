# norax

Neural operators in JAX.

`norax` provides implementations of neural operators built on top of 
[JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox). 

Currently only Fourier neural operators are provided.

## Installation

```bash
pip install norax
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add norax
```

## Quick start

```python
import jax
import jax.numpy as jnp
import norax as nrx

key = jax.random.key(0)

model = nrx.FNO(
    key,
    channels_in=2,
    channels_out=1,
    n_modes=(16,),  # Number of Fourier modes to retain along the spatial axis
    width=64,
    depth=4,
)

# Forward pass over a batch using vmap
x = jnp.ones((10, 256, 2))  # (batch, *grid_shape, channels_in)
y = jax.vmap(model)(x)      # (batch, *grid_shape, channels_out)
```

## Related projects

- [neuraloperator](https://github.com/neuraloperator/neuraloperator): PyTorch implementations of neural operators

## References

Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).