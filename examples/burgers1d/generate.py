"""Generate a dataset of 1D Burgers' equation input-output pairs.

The 1D viscous Burgers' equation is

$$ \\frac{\\partial u}{\\partial t}
= -u \\frac{\\partial u}{\\partial x}
+ \\nu \\frac{\\partial^2 u}{\\partial x^2},
$$
where $\\nu$ is the kinematic viscosity.

The equation has a non-linear advection term $-u \\frac{\\partial u}{\\partial x}$
and a linear diffusion term $\\nu \\frac{\\partial^2 u}{\\partial x^2}$).

We solve on a periodic domain $x \\in [0, 1)$ with initial condition
$u_0(x)$ sampled as
$u_0 \\sim \\mu,$ where $$\\mu = \\mathcal{N}(0,625(-\\Delta + 25I)^{-2}).$$

We discretise Burgers' equation on a uniform periodic grid with $n$ points
and spacing $\\Delta x = L / n$. Both the Laplacian and the first derivative
use second-order accurate central finite differences.

To run this file, users should have h5py installed. This can be done by running
```bash
uv sync --extra examples
```

Usage:

```bash
uv run examples/burgers1d/generate.py \
    --num-samples 1280 \
    --resolution 256 \
    --output examples/burgers1d/data/burgers1d.h5
```
"""

import argparse
import math

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pardax as pdx


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
    """nu * d²u/dx² (periodic, central differences)."""
    nu, dx = params["nu"], params["dx"]
    return nu * (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / dx**2


def advection(t: float, u: jax.Array, params: dict) -> jax.Array:
    """-u * du/dx (periodic, central differences)."""
    dx = params["dx"]
    dudx = (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2 * dx)
    return -u * dudx


def build_stepper(
    nu: float, resolution: int, L: float
) -> tuple[pdx.RK4, pdx.BackwardEuler]:
    """Return explicit (RK4) and implicit (BackwardEuler) time-steppers.

    The pseudo-spectral approach diagonalises the discrete Laplacian to
    advance the diffusive term exactly via pointwise division in Fourier
    space. For the second-order central difference stencil on a periodic
    grid, the eigenvalues of the Laplacian are
    $$ \\sigma_k = \\frac{-4 \\sin^2(k \\Delta x / 2)}{\\Delta x^2}, $$
    where $k = 2\\pi m / L$ are the discrete wavenumbers.
    """
    dx = L / resolution
    k = 2 * jnp.pi * jnp.fft.rfftfreq(resolution, d=dx)
    sigma = -4 * nu * jnp.sin(k * dx / 2) ** 2 / dx**2

    operator = pdx.SpectralOperator(eigvals=sigma)
    spectral_solver = pdx.SpectralSolver(
        forward=jnp.fft.rfft,
        backward=lambda x: jnp.fft.irfft(x, n=resolution),
    )
    root_finder = pdx.LinearRootFinder(
        linsolver=spectral_solver, operator=operator
    )
    return pdx.RK4(), pdx.BackwardEuler(root_finder=root_finder)


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
        "--output",
        type=str,
        required=True,
        help="Path to the output HDF5 file (e.g. data/burgers1d.h5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Samples per batch (controls peak RAM). Default: 64.",
    )
    parser.add_argument("--nu", type=float, default=0.02, help="Viscosity.")
    parser.add_argument("--dt", type=float, default=1e-4, help="Time step.")
    parser.add_argument("--t-end", type=float, default=1.0, help="End time.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n = args.resolution
    L = 1.0
    dx = L / n
    params = {"nu": args.nu, "dx": dx}

    num_steps = math.ceil(args.t_end / args.dt)
    step_size = jnp.asarray(args.t_end / num_steps)

    explicit, implicit = build_stepper(args.nu, n, L)
    grf = GaussianField1D(n, L)
    sample_batch = jax.vmap(jax.jit(lambda key_: grf.sample(key_)))

    def imex_step(carry, _):
        t, y, exp_st, imp_st = carry
        y_star, exp_st = exp_st(advection, t, y, step_size, params)
        y_new, imp_st = imp_st(diffusion, t, y_star, step_size, params)
        return (t + step_size, y_new, exp_st, imp_st), y_new

    @jax.jit
    def solve(y0):
        (_, y_final, _, _), _ = jax.lax.scan(
            imex_step, (0.0, y0, explicit, implicit), length=num_steps
        )
        return y_final

    solve_batch = jax.vmap(solve)

    key = jax.random.key(args.seed)

    # Infer dtype from a single sample
    key, subkey = jax.random.split(key)
    _y0_probe = grf.sample(subkey)
    dtype = np.asarray(_y0_probe).dtype

    x = np.linspace(0, L, num=n, endpoint=False, dtype=dtype)

    with h5py.File(args.output, "w") as f:
        ds_in = f.create_dataset(
            "inputs",
            shape=(args.num_samples, n, 2),
            dtype=dtype,
            chunks=(min(args.batch_size, args.num_samples), n, 2),
        )
        ds_out = f.create_dataset(
            "outputs",
            shape=(args.num_samples, n, 1),
            dtype=dtype,
            chunks=(min(args.batch_size, args.num_samples), n, 1),
        )

        # Store run metadata as attributes
        ds_in.attrs["description"] = (
            "[sample, :, 0] = grid points x, "
            "[sample, :, 1] = initial condition u_0(x)"
        )
        ds_out.attrs["description"] = "[sample, :, 0] = solution u(x, t_end)"
        f.attrs.update(
            {
                "nu": args.nu,
                "dt": args.dt,
                "t_end": args.t_end,
                "resolution": n,
                "num_samples": args.num_samples,
                "L": L,
            }
        )

        for start in range(0, args.num_samples, args.batch_size):
            end = min(start + args.batch_size, args.num_samples)
            batch = end - start

            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, batch)
            y0 = sample_batch(batch_keys)  # (batch, n)

            y_end = solve_batch(y0)  # (batch, n)

            inputs_np = np.empty((batch, n, 2), dtype=dtype)
            inputs_np[:, :, 0] = x[None, :]
            inputs_np[:, :, 1] = np.asarray(y0)

            ds_in[start:end] = inputs_np
            ds_out[start:end] = np.asarray(y_end)[:, :, None]

            print(f"  {end}/{args.num_samples} samples written")

    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
