# Burgers' equation in one dimension

This file describes how to use the scripts in this directory to 1D to
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
'examples' dependency group installed. This can be done by running:

```bash
uv sync --extra examples
```

## 1. Data Generation

This section describes how the training dataset for the 1D Burgers' equation
example is generated. The source code is in `examples/burgers1d/generate.py`.

### 1.1 Initial conditions

Initial conditions are drawn from a Gaussian random field:

$$ u_0 \sim \mathcal{N}\!\left(0,\; 625(-\Delta + 25I)^{-2}\right), $$

where $\Delta$ is the Laplacian on the periodic domain.

### 1.2 Discretisation

The spatial derivatives are discretised using second-order accurate central
differences.

The solution is advanced through time with an implicit-explicit (IMEX) scheme.
The advective term is advanced with a forward Euler method and the diffusive
term is advanced with a backward Euler method, where the linear system
is diagonalised and inverted directly in Fourier space.

For further details on numerical time integration in JAX, check out 
[pardax](https://github.com/christianfenton/pardax).

### 1.3 Dataset format

The dataset is stored in an HDF5 file with two datasets:
- `inputs` with shape `(N, n, 2)`
- `outputs` with shape `(N, n, 1)`

For each sample $i$, the grid points are `x = inputs[i, :, 0]` 
and the initial condition is `u0 = inputs[i, :, 1]`

Metadata are stored as HDF5 attributes on the root group.

### 1.4 Generating the dataset

Install the examples dependencies if they're not already installed:

```bash
uv sync --extra examples
```

Then run the generation script:

```bash
uv run examples/burgers1d/generate_burgers1d.py \
    --num-samples 1280 \
    --resolution 2048 \
    --output examples/burgers1d/data/burgers1d.h5
```

**CLI Options:**

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--num-samples` | *required* | Total number of samples to generate |
| `--resolution`| *required* | Number of spatial grid points |
| `--output` | *required* | Path to the output HDF5 file |
| `--nu` | `0.02` | Viscosity |
| `--dt` | `1e-4` | Time step size |
| `--t_end` `1.0` | | End time |
| `--batch-size` | `64`  | Samples generated per batch |
| `--seed` | `0` | Seed for pseudo-random number generator

## 2. Training

This section describes how to train an FNO on the 1D Burgers' equation 
dataset and visualise the results.

Source files:

- `examples/burgers1d/train.py`: training script
- `examples/burgers1d/visualise.ipynb`: visualisation notebook


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
uv run examples/burgers1d/train_burgers1d.py \
    --data examples/burgers1d/data/burgers1d.h5 \
    --output examples/burgers1d/models/burgers1d_fno_256.eqx \
    --resolution 256 \
    --n-train 1024 --n-test 256
```

**CLI options**

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Path to the HDF5 dataset |
| `--output` | *(required)* | Path to save the trained model |
| `--n-train` | *(required)* | Number of training samples |
| `--n-test` | *(required)* | Number of test samples |
| `--resolution` | `None` | Target resolution after downsampling |
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

After training, open the notebook to visualise the predictions:

```bash
uv run jupyter notebook examples/burgers1d/visualise.ipynb
```