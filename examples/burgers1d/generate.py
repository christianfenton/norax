"""Generate a dataset of 1D Burgers' equation input-output pairs.

The 1D viscous Burgers' equation is

du/dt = -u du/dx + nu divgrad(u),

where nu is the kinematic viscosity, and du/dt and du/dx are partial
derivatives w.r.t. the temporal and spatial coordinates, respectively.

The (non-linear) advective term is -u du/dx.
The (linear) diffusive term is nu divgrad(u).

We solve on a periodic spatial domain [0, 1)
with initial condition u_0(x) := u(t=0, x) sampled from a Gaussian with
mean zero and covariance matrix 625(-Delta + 25 I)^2,
where I is the identity matrix and Delta represents the Laplacian (i.e. divgrad).
In frequency space, the Laplacian is diagonal with eigenvalues -(1/k)^2,
so the initial conditions can be easily generated in the corresponding basis.

The field u is discretised on a uniform spatial grid with n points
spaced by 1 / n. The derivatives are approximated with second-order accurate
central finite differences. The discretisation of the Laplacian leads to
different eigenvalues to the continuous case, and are given by:

-4 * sin(k * dx / 2) ** 2 / dx**2,

where dx is the grid spacing and k are the discrete wavenumbers.

The convective term is advanced with a fourth-order Runge-Kutta (RK4) method,
while the diffusive term is discretised with a backward Euler method and the
resulting linear system is solved exactly by a point-wise multiplication in
spectral space.

To run this file, users should have pyarrow installed. This can be done by
running:
```bash
uv sync --extra examples
```

Usage:

```bash
uv run examples/burgers1d/generate.py \
    --num-samples 1280 \
    --resolution 256 \
    --output-dir examples/burgers1d/data
```
"""

import argparse
import json
import math
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pardax as pdx
import pyarrow as pa
import pyarrow.parquet as pq


class GaussianField1D(eqx.Module):
    """Sample 1D fields from N(0, 625(-Δ + 25I)^{-2}) on a periodic grid."""

    coef: jax.Array
    k: jax.Array
    n: int
    L: float

    def __init__(self, n: int, L: float) -> None:
        self.n = n
        self.L = L
        k = 2 * jnp.pi * jnp.fft.fftfreq(n, d=L / n)
        self.k = k
        self.coef = jnp.sqrt(625 * (k**2 + 25) ** (-2))

    def sample(self, key: jax.Array) -> jax.Array:
        n = self.n
        coef = self.coef

        key_dc, key_re, key_im, key_nyq = jax.random.split(key, 4)
        dc = jax.random.normal(key_dc, (1,))

        m = (n - 1) // 2
        re_int = jax.random.normal(key_re, (m,))
        im_int = jax.random.normal(key_im, (m,))
        interior_modes = (re_int + 1j * im_int) / jnp.sqrt(2.0)

        if n % 2 == 0:
            nyquist = jax.random.normal(key_nyq, (1,))
            pos = jnp.concatenate([dc, interior_modes, nyquist])
        else:
            pos = jnp.concatenate([dc, interior_modes])

        neg = jnp.conj(jnp.flip(pos[1 : m + 1]))
        noise = jnp.concatenate([pos, neg])

        f_k = coef * noise
        f_x = jnp.fft.ifft(f_k) * n
        return jnp.real(f_x)


def diffusion(t: float, u: jax.Array, params: dict) -> jax.Array:
    nu, dx = params["nu"], params["dx"]
    return nu * (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / dx**2


def advection(t: float, u: jax.Array, params: dict) -> jax.Array:
    dx = params["dx"]
    dudx = (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2 * dx)
    return -u * dudx


def build_steppers(
    nu: float, resolution: int, L: float
) -> tuple[pdx.RK4, pdx.BackwardEuler]:
    """Return RK4 and BackwardEuler time-steppers."""

    # Eigenvalues of the discretised Laplacian
    dx = L / resolution
    k = 2 * jnp.pi * jnp.fft.rfftfreq(resolution, d=dx)
    sigma = -4 * nu * jnp.sin(k * dx / 2) ** 2 / dx**2
    operator = pdx.SpectralOperator(eigvals=sigma)

    # Define the transformations to spectral space
    spectral_solver = pdx.SpectralSolver(
        forward=jnp.fft.rfft,
        backward=lambda x: jnp.fft.irfft(x, n=resolution),
    )

    root_finder = pdx.LinearRootFinder(
        linsolver=spectral_solver, operator=operator
    )

    backward_euler_stepper = pdx.BackwardEuler(root_finder=root_finder)

    rk4_stepper = pdx.RK4()

    return rk4_stepper, backward_euler_stepper


def imex_step(carry, _):
    """
    Take a single implicit-explicit (imex) step.

    This function is intended as the function argument in `jax.lax.scan`.

    Carry is a tuple of (t, y, exp_st, imp_st, step_size, params), where
        t: Time
        y: State (i.e. the field u in Burgers' equation)
        exp_st: The explicit time-stepper
        imp_st: The implicit time-stepper
        step_size: The size of the step taken
        params: A dict of additional parameters
    """
    t, y, exp_st, imp_st, step_size, params = carry
    y_star, exp_st = exp_st(advection, t, y, step_size, params)
    y_new, imp_st = imp_st(diffusion, t, y_star, step_size, params)
    return (t + step_size, y_new, exp_st, imp_st, step_size, params), None


def make_dataset_name(nu: float, resolution: int) -> str:
    nu_str = str(nu).replace(".", "p")
    return f"burgers1d_nu{nu_str}_res{resolution}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 1D Burgers' equation dataset."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Total number of samples to generate.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="Number of spatial grid points.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Parent directory for the output dataset folder "
            "(e.g. examples/burgers1d/data)."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Dataset directory name. Defaults to "
            "burgers1d_nu<nu>_res<resolution>."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Samples per batch (controls peak JAX memory). Default: 64.",
    )
    parser.add_argument("--nu", type=float, default=0.02, help="Viscosity.")
    parser.add_argument("--dt", type=float, default=1e-4, help="Time step.")
    parser.add_argument("--t-end", type=float, default=1.0, help="End time.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Floating-point precision of stored arrays. Default: float32.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    dtype = np.dtype(args.dtype)

    key = jax.random.key(args.seed)
    n = args.resolution
    L = 1.0
    dx = L / n
    params = {"nu": args.nu, "dx": dx}

    grf = GaussianField1D(n, L)
    sample_batch = jax.vmap(jax.jit(lambda key_: grf.sample(key_)))

    x = np.linspace(0, L, num=n, endpoint=False, dtype=dtype)

    num_steps = math.ceil(args.t_end / args.dt)
    step_size = jnp.asarray(args.t_end / num_steps)
    exp_st, imp_st = build_steppers(args.nu, n, L)

    @jax.jit
    def solve(y0):
        carry, _ = jax.lax.scan(
            imex_step,
            (0.0, y0, exp_st, imp_st, step_size, params),
            length=num_steps,
        )
        y_final = carry[1]
        return y_final

    solve_batch = jax.vmap(solve)

    pa_float = pa.from_numpy_dtype(dtype)
    schema = pa.schema(
        [
            pa.field("sample_id", pa.int64()),
            pa.field("x", pa.list_(pa_float)),
            pa.field("u0", pa.list_(pa_float)),
            pa.field("u_end", pa.list_(pa_float)),
        ]
    )

    dataset_name = args.name or make_dataset_name(args.nu, n)
    dataset_dir = pathlib.Path(args.output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with pq.ParquetWriter(
        dataset_dir / "data.parquet", schema, compression="snappy"
    ) as writer:
        for start in range(0, args.num_samples, args.batch_size):
            end = min(start + args.batch_size, args.num_samples)
            batch = end - start

            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, batch)
            y0 = sample_batch(batch_keys)  # (batch, n)
            y_end = solve_batch(y0)  # (batch, n)

            y0_np = np.asarray(y0, dtype=dtype)
            y_end_np = np.asarray(y_end, dtype=dtype)
            sample_ids = np.arange(start, end, dtype=np.int64)
            # PyArrow list columns are stored as a flat values buffer plus an
            # offsets array where offsets[i] is the start of row i, so row i
            # spans flat_values[offsets[i]:offsets[i+1]].
            offsets = np.arange(0, (batch + 1) * n, n, dtype=np.int32)

            rb = pa.RecordBatch.from_arrays(
                [
                    pa.array(sample_ids, type=pa.int64()),
                    pa.ListArray.from_arrays(
                        pa.array(offsets, type=pa.int32()),
                        pa.array(np.tile(x, batch), type=pa_float),
                    ),
                    pa.ListArray.from_arrays(
                        pa.array(offsets, type=pa.int32()),
                        pa.array(y0_np.flatten(), type=pa_float),
                    ),
                    pa.ListArray.from_arrays(
                        pa.array(offsets, type=pa.int32()),
                        pa.array(y_end_np.flatten(), type=pa_float),
                    ),
                ],
                schema=schema,
            )
            writer.write_batch(rb)

            print(f"  {end}/{args.num_samples} samples generated")

    metadata = {
        "nu": args.nu,
        "dt": args.dt,
        "t_end": args.t_end,
        "resolution": n,
        "L": L,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "dtype": args.dtype,
    }
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Dataset saved to {dataset_dir}")


if __name__ == "__main__":
    main()
