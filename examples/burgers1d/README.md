# Burgers' equation in one dimension

This file describes how to use the scripts in this directory to
generate a dataset from Burgers' equation, train an FNO on it,
and visualise the results.

The 1D viscous Burgers' equation on a periodic domain $x \in [0, 1)$ reads

$$
\frac{\partial u}{\partial t}
= -u\frac{\partial u}{\partial x} + \nu\frac{\partial^2 u}{\partial x^2},
$$

where $\nu > 0$ is the kinematic viscosity. The first term is a non-linear
advective term and the second is a diffusive term.

The learning task is to approximate the solution operator
$$ \mathcal{G}^\dagger : u_0 \mapsto u(\cdot,\, t_\text{end}), $$
mapping an initial condition $u_0$ to the solution at time $t_\text{end}$.

**Note:** To run the scripts in this directory, users need to have the
`examples` dependencies installed:

```bash
uv sync --extra examples
```

## 1. Data Generation

Source file: `examples/burgers1d/generate.py`

### 1.1 Initial conditions

Initial conditions are drawn from a Gaussian random field:

$$ u_0 \sim \mathcal{N}\!\left(0,\; 625(-\Delta + 25I)^{-2}\right), $$

where $\Delta$ is the Laplacian on the periodic domain.

### 1.2 Discretisation

The spatial derivatives are discretised using second-order accurate central
differences.

The solution is advanced through time with an implicit-explicit (IMEX) scheme.
The advective term is advanced with a fourth-order Runge-Kutta (RK4) method
and the diffusive term is advanced with a backward Euler method, where the
linear system is inverted directly in Fourier space.

For further details on numerical time integration in JAX, check out
[pardax](https://github.com/christianfenton/pardax).

### 1.3 Data schema

The dataset is stored as a directory containing:
- `data.parquet`: Columnar data file
- `metadata.json`: Dataset attributes

The columns in `data.parquet` are:
- `sample_id` (`int64`): Sample index
- `x` (`list<dtype>`, length `n`): Spatial grid coordinates
- `u0` (`list<dtype>`, length `n`): Initial condition $u_0(x)$
- `u_end` (`list<dtype>`, length `n`): Solution $u(x, t_\text{end})$

The attributes in `metadata.json` are:
- `nu`: Viscosity
- `dt`: Time step size
- `t_end`: End time
- `resolution`: Number of grid points
- `L`: Length of the domain
- `seed`: Seed for pseudo-random number generator
- `num_samples`: Number of samples (rows) in the data file
- `dtype`: Data type

### 1.4 Generating the dataset

```bash
uv run examples/burgers1d/generate.py \
    --num-samples 1280 \
    --resolution 2048 \
    --output-dir examples/burgers1d/data
```

This creates `examples/burgers1d/data/burgers1d_nu0p02_res2048/`.

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-samples` | *(required)* | Total number of samples to generate |
| `--resolution` | *(required)* | Number of spatial grid points |
| `--output-dir` | *(required)* | Parent directory for the output dataset folder |
| `--name` | auto | Dataset directory name (default: `burgers1d_nu<nu>_res<resolution>`) |
| `--nu` | `0.02` | Viscosity |
| `--dt` | `1e-4` | Time step size |
| `--t-end` | `1.0` | End time |
| `--batch-size` | `64` | Samples generated per batch |
| `--dtype` | `float32` | Floating-point precision (`float32` or `float64`) |
| `--seed` | `0` | Seed for pseudo-random number generator |

## 2. Training

Source file: `examples/burgers1d/train.py`

### 2.1 Model

An FNO with `channels_in=2` (spatial coordinate and initial condition)
and `channels_out=1` (solution at $t_\text{end}$) is trained to
approximate the solution operator
$\mathcal{G}^\dagger : u_0 \mapsto u(\cdot,\,t_\text{end})$.

### 2.2 Cost function

Training minimises the mean relative $L^2$ error over a mini-batch:

$$
\mathcal{C}
= \frac{1}{B}\sum_{i=1}^{B} \frac{\|\hat{u}_i - u_i\|_2}{\|u_i\|_2},
$$

where $\hat{u}_i$ is the model prediction and
$u_i$ is the ground-truth solution for sample $i$.

### 2.3 Optimisation

The model is trained with Adam and a learning rate that halves every 100 epochs.
The default initial learning rate is `1e-3`.

### 2.4 Training the model

```bash
uv run examples/burgers1d/train.py \
    --data examples/burgers1d/data/burgers1d_nu0p02_res2048 \
    --output examples/burgers1d/models/burgers1d_fno_256.eqx \
    --resolution 256 \
    --n-train 1024 --n-test 256
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Path to the dataset directory produced by `generate.py` |
| `--output` | *(required)* | Path to save the trained model |
| `--n-train` | *(required)* | Number of training samples |
| `--n-test` | *(required)* | Number of test samples |
| `--resolution` | `None` | Target resolution after downsampling (must divide dataset resolution) |
| `--n-modes` | `16` | Maximum Fourier modes per axis |
| `--width` | `64` | Hidden channel width |
| `--depth` | `4` | Number of Fourier layers |
| `--n-epochs` | `500` | Training epochs |
| `--batch-size` | `32` | Samples per mini-batch |
| `--lr` | `1e-3` | Initial Adam learning rate |
| `--seed` | `0` | PRNG seed |

### 2.5 Saving and loading models

The training script serialises the model as a binary file with a
JSON header containing the constructor hyperparameters:

```python
save_model("model.eqx", model, hyperparams)

model, hyperparams = load_model("model.eqx")
```

Both functions are defined in `train.py` and follow Equinox's
recommended serialisation pattern.

## 3. Visualisation

Source files:

- `examples/burgers1d/visualise.py`: GIF of the PDE solution and FNO predictions for random initial conditions
- `examples/burgers1d/visualise_sine.py`: FNO prediction for a sinusoidal initial condition

Both scripts write their outputs to `examples/burgers1d/figures/`.

```bash
uv run examples/burgers1d/visualise.py
uv run examples/burgers1d/visualise_sine.py
```
