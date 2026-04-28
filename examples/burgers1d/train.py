"""Train a Fourier neural operator (FNO) on the 1D Burgers' equation dataset.

This script:
1. Loads a Parquet dataset produced by ``generate.py``
2. (Optionally) Down-samples the data to a coarser grid
3. Trains an FNO to learn the mapping from initial condition to final solution
4. Saves (serialises) the trained model to disk.

Usage:

```bash
uv run examples/burgers1d/train.py \
    --data examples/burgers1d/data/burgers1d_nu0p2_res2048 \
    --output examples/burgers1d/models/burgers1d_fno_256.eqx \
    --resolution 256 \
    --n-train 1024 --n-test 256
```
"""

import argparse
import json
import math
import pathlib
from functools import partial

import datasets
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import install_import_hook

with install_import_hook("norax", "beartype.beartype"):
    from norax.models import FNO


def load_parquet(data_dir: str) -> tuple[datasets.Dataset, dict]:
    """Load a Parquet dataset directory produced by ``generate.py``.

    Args:
        data_dir: Path to a dataset directory. Must contain ``metadata.json``
            and ``data.parquet``.

    Returns:
        dataset: HuggingFace Dataset with columns ``sample_id``, ``x``,
            ``u0``, and ``u_end``.
        metadata: Dictionary of dataset attributes.
    """
    root = pathlib.Path(data_dir)
    with open(root / "metadata.json") as f:
        metadata = json.load(f)

    n = metadata["resolution"]
    features = datasets.Features(
        {
            "sample_id": datasets.Value("int64"),
            "x": datasets.Sequence(datasets.Value("float32"), length=n),
            "u0": datasets.Sequence(datasets.Value("float32"), length=n),
            "u_end": datasets.Sequence(datasets.Value("float32"), length=n),
        }
    )
    dataset = datasets.load_dataset(
        "parquet",
        data_files=str(root / "data.parquet"),
        split="train",
        features=features,
    )
    return dataset, metadata


def preprocess(
    dataset: datasets.Dataset, n: int, stride: int = 1
) -> datasets.Dataset:
    """Build ``input`` and ``output`` columns from raw Parquet columns.

    Optionally subsamples the spatial grid by ``stride``. The resulting
    ``input`` column stacks the spatial grid and initial condition along the
    channel axis; ``output`` wraps the solution in a trailing size-1 channel.

    Args:
        dataset: Raw dataset with columns ``x``, ``u0``, and ``u_end``.
        n: Original spatial resolution (number of grid points).
        stride: Subsampling stride. ``1`` keeps the original resolution.

    Returns:
        Dataset with columns ``input`` of shape ``(n // stride, 2)`` and
        ``output`` of shape ``(n // stride, 1)`` per sample.
    """
    target_n = n // stride
    out_features = datasets.Features(
        {
            "input": datasets.Array2D(shape=(target_n, 2), dtype="float32"),
            "output": datasets.Array2D(shape=(target_n, 1), dtype="float32"),
        }
    )

    def _build(batch):
        x = jnp.array(batch["x"])[:, ::stride]
        u0 = jnp.array(batch["u0"])[:, ::stride]
        u_end = jnp.array(batch["u_end"])[:, ::stride]
        return {
            "input": jnp.stack([x, u0], axis=-1),
            "output": u_end[:, :, None],
        }

    return dataset.map(
        _build,
        batched=True,
        remove_columns=dataset.column_names,
        features=out_features,
    )


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


def evaluate(model: FNO, dataset: datasets.Dataset, batch_size: int) -> float:
    total_loss = jnp.zeros(())
    n_samples = 0
    for batch in dataset.iter(batch_size=batch_size):
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
    dataset: datasets.Dataset,
    batch_size: int,
) -> tuple[FNO, optax.OptState, float]:
    total_loss = jnp.zeros(())
    n_samples = 0
    for batch in dataset.iter(batch_size=batch_size):
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
        help="Path to Parquet dataset directory produced by generate.py.",
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

    # Load and preprocess data
    raw_ds, metadata = load_parquet(args.data)
    orig_res = metadata["resolution"]
    print(
        f"Loaded {len(raw_ds)} samples at resolution {orig_res} "
        f"(nu={metadata.get('nu')}, t_end={metadata.get('t_end')})"
    )

    if args.resolution is not None and args.resolution != orig_res:
        if orig_res % args.resolution != 0:
            raise ValueError(
                f"Original resolution {orig_res} is not evenly divisible by "
                f"target resolution {args.resolution}."
            )
        stride = orig_res // args.resolution
        print(f"Down-sampling to resolution {args.resolution}")
    else:
        stride = 1

    total_needed = args.n_train + args.n_test
    if total_needed > len(raw_ds):
        raise ValueError(
            f"Dataset has {len(raw_ds)} samples but "
            f"n_train + n_test = {total_needed}."
        )

    ds = preprocess(
        raw_ds.select(range(total_needed)), orig_res, stride
    ).with_format("jax")

    splits = ds.train_test_split(test_size=args.n_test, shuffle=False)
    train_ds = splits["train"]
    test_ds = splits["test"]

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
    steps_per_epoch = math.ceil(args.n_train / args.batch_size)
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
        shuffled = train_ds.shuffle(seed=args.seed + epoch)
        model, opt_state, train_loss = run_epoch(
            model, opt_state, optimiser, shuffled, args.batch_size
        )

        if epoch % 10 == 0 or epoch == args.n_epochs:
            test_loss = evaluate(model, test_ds, args.batch_size)
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
