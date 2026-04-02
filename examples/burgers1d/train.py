"""Train a Fourier neural operator (FNO) on the 1D Burgers' equation dataset.

The script loads an HDF5 dataset produced by ``generate_burgers1d.py``,
optionally down-resolves it to a coarser grid, trains an FNO to learn the
operator mapping the initial condition to the solution at time ``t_end``,
and serialises the trained model to disk.

Usage:

```bash
uv run examples/burgers1d/train.py \
    --data examples/burgers1d/data/burgers1d.h5 \
    --output examples/burgers1d/models/burgers1d_fno_256.eqx \
    --resolution 256 \
    --n-train 1024 --n-test 256
```
"""

import argparse
import json
from functools import partial

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import install_import_hook

with install_import_hook("norax", "beartype.beartype"):
    from norax.data import DataLoader
    from norax.models import FNO


def load_data(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load inputs, outputs, and metadata from an HDF5 dataset file.

    Args:
        path: Path to an HDF5 file produced by ``generate_burgers1d.py``.

    Returns:
        inputs: Array of shape ``(N, resolution, 2)`` where channel 0 is the
            spatial grid and channel 1 is the initial condition ``u_0(x)``.
        outputs: Array of shape ``(N, resolution, 1)`` containing the
            solution ``u(x, t_end)``.
        metadata: Dictionary of dataset attributes
    """
    with h5py.File(path, "r") as f:
        inputs = np.asarray(f["inputs"])
        outputs = np.asarray(f["outputs"])
        metadata = dict(f.attrs)
    return inputs, outputs, metadata


def downsample(
    inputs: np.ndarray,
    outputs: np.ndarray,
    target_res: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample inputs and outputs to a coarser spatial resolution.

    Args:
        inputs: Array of shape ``(N, res, 2)``
        outputs: Array of shape ``(N, res, 1)``
        target_res: Desired number of grid points after subsampling

    Returns:
        Downsampled inputs and outputs
    """
    orig_res = inputs.shape[1]
    if orig_res % target_res != 0:
        raise ValueError(
            f"Original resolution {orig_res} is not evenly divisible by "
            f"target resolution {target_res}."
        )
    stride = orig_res // target_res
    return inputs[:, ::stride, :], outputs[:, ::stride, :]


def save_model(path: str, model: FNO, hyperparams: dict) -> None:
    """Save model weights and hyperparameters to a binary file.

    The file format follows the Equinox recommended pattern:
    a JSON-encoded hyperparameter line followed by the serialised leaves.

    Args:
        path: Destination file path (e.g. ``data/model.eqx``).
        model: Trained FNO instance.
        hyperparams: Dictionary of constructor arguments needed to recreate
            the model architecture (``channels_in``, ``channels_out``,
            ``n_modes``, ``width``, ``depth``).
    """
    with open(path, "wb") as f:
        f.write((json.dumps(hyperparams) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(path: str):
    """Load a model previously saved with `save_model`.

    Args:
        path: Path to the saved model file.

    Returns:
        model: FNO instance with restored weights.
        hyperparams: Hyperparameter dictionary read from the file.
    """
    with open(path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        hyperparams["n_modes"] = tuple(hyperparams["n_modes"])
        skeleton = FNO(key=jax.random.key(0), **hyperparams)
        model = eqx.tree_deserialise_leaves(f, skeleton)
    return model, hyperparams


def relative_l2(prediction: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.linalg.norm(prediction - target) / jnp.linalg.norm(target)


@partial(jax.vmap, in_axes=(None, 0, 0))
def loss_fn(model: FNO, x: jax.Array, y: jax.Array) -> jax.Array:
    return relative_l2(model(x), y)


@jax.jit
def eval_step(model: FNO, x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.sum(loss_fn(model, x, y))


def evaluate(model: FNO, loader: DataLoader) -> float:
    total_loss = jnp.zeros(())
    n_samples = 0
    for batch in loader:
        x, y = batch["input"], batch["output"]
        n_samples += x.shape[0]
        total_loss += eval_step(model, x, y)
    return float(total_loss / n_samples)


@eqx.filter_jit
def train_step(
    model: FNO,
    opt_state: optax.OptState,
    optimiser: optax.GradientTransformation,
    x: jax.Array,
    y: jax.Array,
) -> tuple[FNO, optax.OptState, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(
        lambda m: jnp.mean(loss_fn(m, x, y))
    )(model)
    updates, opt_state = optimiser.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss * x.shape[0]


def run_epoch(
    model: FNO,
    opt_state: optax.OptState,
    optimiser: optax.GradientTransformation,
    loader: DataLoader,
) -> tuple[FNO, optax.OptState, float]:
    total_loss = jnp.zeros(())
    n_samples = 0
    for batch in loader:
        x, y = batch["input"], batch["output"]
        n_samples += x.shape[0]
        model, opt_state, batch_loss = train_step(
            model, opt_state, optimiser, x, y
        )
        total_loss += batch_loss
    return model, opt_state, float(total_loss / n_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an FNO on the 1D Burgers' equation dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the HDF5 dataset produced by generate_burgers1d.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "Target spatial resolution after down-sampling. "
            "Must evenly divide the original dataset resolution. "
            "Defaults to the original resolution."
        ),
    )
    parser.add_argument(
        "--n-train",
        type=int,
        required=True,
        help="Number of training samples.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        required=True,
        help="Number of test samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Samples per batch. Default: 32.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=16,
        help="Maximum number of Fourier modes retained per axis. Default: 16.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Hidden channel width. Default: 64.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of Fourier layers. Default: 4.",
    )
    parser.add_argument(
        "--project-hidden-dims",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help=(
            "Hidden layer widths of the projection MLP. "
            "Defaults to a single hidden layer of size --width."
        ),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=500,
        help="Number of training epochs. Default: 500.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam. Default: 1e-3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed. Default: 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    inputs, outputs, metadata = load_data(args.data)
    orig_res = inputs.shape[1]
    print(
        f"Loaded {inputs.shape[0]} samples at resolution {orig_res} "
        f"(nu={metadata.get('nu')}, t_end={metadata.get('t_end')})"
    )

    # Optional down-resolution
    if args.resolution is not None and args.resolution != orig_res:
        inputs, outputs = downsample(inputs, outputs, args.resolution)
        resolution = args.resolution
        print(f"Down-sampled to resolution {resolution}")
    else:
        resolution = orig_res

    total_needed = args.n_train + args.n_test
    if total_needed > inputs.shape[0]:
        raise ValueError(
            f"Dataset has {inputs.shape[0]} samples but "
            f"n_train + n_test = {total_needed}."
        )

    # Build DataLoaders
    inputs_jax = jnp.asarray(inputs[:total_needed])
    outputs_jax = jnp.asarray(outputs[:total_needed])

    train_data = {
        "input": inputs_jax[: args.n_train],
        "output": outputs_jax[: args.n_train],
    }
    test_data = {
        "input": inputs_jax[args.n_train : total_needed],
        "output": outputs_jax[args.n_train : total_needed],
    }

    train_loader = DataLoader(
        train_data, args.batch_size, shuffle=True, seed=args.seed
    )
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False, seed=0)

    # Model
    hyperparams = {
        "channels_in": 2,
        "channels_out": 1,
        "n_modes": (args.n_modes,),
        "width": args.width,
        "depth": args.depth,
    }
    key = jax.random.key(args.seed)
    key, subkey = jax.random.split(key)
    model = FNO(key=subkey, **hyperparams)
    print(
        f"FNO: n_modes={args.n_modes}, width={args.width}, depth={args.depth}, "
    )

    # Optimiser
    steps_per_epoch = len(train_loader)
    schedule = optax.schedules.exponential_decay(
        args.lr,
        transition_steps=steps_per_epoch * 100,
        decay_rate=0.5,
        staircase=True,
    )
    optimiser = optax.adam(schedule)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

    # Training loop
    for epoch in range(1, args.n_epochs + 1):
        model, opt_state, train_loss = run_epoch(
            model, opt_state, optimiser, train_loader
        )
        train_loader.reset()

        if epoch % 10 == 0 or epoch == args.n_epochs:
            test_loss = evaluate(model, test_loader)
            test_loader.reset()
            print(
                f"Epoch {epoch:4d}/{args.n_epochs}  "
                f"train={train_loss:.4e}  test={test_loss:.4e}"
            )

    # Save
    save_model(
        args.output,
        model,
        {
            **hyperparams,
            "n_modes": list(hyperparams["n_modes"]),
        },
    )
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
