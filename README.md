# norax

Neural operators in JAX.

`norax` provides implementations of neural operators built on top of 
[JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox). 

Features:

- Fourier neural operators (FNO): learns mappings between function spaces via
  learnable convolutions in the frequency domain
  ([Li et al., 2020](https://arxiv.org/abs/2010.08895))

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
    channels_in=2,   # e.g. [grid coordinates, function values]
    channels_out=1,
    n_modes=(16,),   # Fourier modes to retain per spatial axis
    width=64,
    depth=4,
)

# Forward pass over a batch using vmap
x = jnp.ones((10, 64, 2))  # (batch, resolution, channels_in)
y = jax.vmap(model)(x)     # (batch, resolution, channels_out)
```

The `n_modes` tuple determines the spatial dimensionality of the operator:
a 1-tuple gives a 1D FNO, a 2-tuple gives a 2D FNO, and so on.

## Development

Dependencies are split into groups for different workflows:

| Group | Purpose | Install |
|-------|---------|---------|
| `test` | pytest, pytest-codeblocks, beartype | `uv sync --group test` |
| `docs` | mkdocs and plugins | `uv sync --group docs` |
| `examples` | h5py, matplotlib, ipykernel | `uv sync --extra examples` |
| `lint` | ruff, mypy | `uv sync --group lint` |

To install everything at once:
```bash
uv sync --all-groups --all-extras
```

## Documentation

Build the documentation locally by running
```bash
uv run mkdocs build
```
or serve them as a local webpage with
```bash
uv run mkdocs serve
```