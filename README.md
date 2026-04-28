# norax

Neural operators in JAX.

[![tests](https://github.com/christianfenton/norax/actions/workflows/tests.yml/badge.svg)](https://github.com/christianfenton/norax/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/christianfenton/norax/graph/badge.svg)](https://codecov.io/gh/christianfenton/norax)
[![lint](https://github.com/christianfenton/norax/actions/workflows/lint.yml/badge.svg)](https://github.com/christianfenton/norax/actions/workflows/lint.yml)
[![docs](https://github.com/christianfenton/norax/actions/workflows/docs.yml/badge.svg)](https://github.com/christianfenton/norax/actions/workflows/docs.yml)

`norax` provides implementations of neural operators built on top of 
[JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox).

Neural operators learn mappings between function spaces.

Currently only Fourier neural operators 
([Li et al., 2020](https://arxiv.org/abs/2010.08895)) 
are provided in this library.

Check out the [documentation page](https://christianfenton.github.io/norax)
for more details.

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
from norax.models import FNO

key = jax.random.key(0)

# Define a 1D Fourier neural operator
model = FNO(
    key,
    channels_in=2,   # 1D scalar field: [x-coordinate, field value]
    channels_out=1,
    n_modes=(16,),   # Fourier modes to retain per spatial axis
    width=64,
    depth=4,
)

# Forward pass over a batch using vmap
x = jnp.ones((10, 64, 2))  # (batch, *grid_shape, channels_in)
y = jax.vmap(model)(x)     # (batch, *grid_shape, channels_out)
```

The `n_modes` tuple determines the spatial dimensionality of the operator:
a 1-tuple gives a 1D FNO, a 2-tuple gives a 2D FNO, and so on.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on setting up a 
development environment, running tests, linting, and building the documentation
locally.