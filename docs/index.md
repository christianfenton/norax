# norax

Neural operators in JAX.

`norax` provides implementations of neural operators built on top of 
[JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox). 

Currently only Fourier neural operators are provided.

## Installation

[uv](https://docs.astral.sh/uv/) is recommended for installation.

### Using uv

With SSH:
```bash
uv init my-project
cd my-project
uv add git+ssh://git@github.com/christianfenton/norax.git
```

With HTTPS:
```bash
uv add git+https://github.com/christianfenton/norax.git
```

### Using pip

With SSH:
```bash
pip install git+ssh://git@github.com/christianfenton/norax.git
```

With HTTPS:
```bash
pip install git+https://github.com/christianfenton/norax.git
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