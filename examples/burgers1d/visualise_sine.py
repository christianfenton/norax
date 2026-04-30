"""FNO prediction for initial condition u_0 = sin(2 π x / L).

The training data uses Gaussian random fields as initial conditions,
so this tests the model's ability to generalise to a qualitatively
different input.

Outputs:
    figures/burgers_fno_sine.png: solver vs FNO for the sine IC

Usage:
    uv run examples/burgers1d/visualise_sine.py
"""

import json
import pathlib
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pardax as pdx
from generate import advection, build_steppers, diffusion

import norax as nrx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = pathlib.Path(__file__).parent / "models" / "burgers1d_fno_256.eqx"
OUTPUT_DIR = pathlib.Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

N = 256
L = 1.0
NU = 0.02
T_END = 1.0
STEP_SIZE = 1e-4
STEPS_PER_FRAME = 100
N_FRAMES = int(T_END / (STEP_SIZE * STEPS_PER_FRAME))

DX = L / N
PARAMS = {"nu": NU, "dx": DX}

# ---------------------------------------------------------------------------
# Helpers (copied from visualise.py)
# ---------------------------------------------------------------------------


def load_model(path: pathlib.Path):
    with open(path, "rb") as f:
        hp = json.loads(f.readline().decode())
        hp["n_modes"] = tuple(hp["n_modes"])
        skeleton = nrx.FNO(key=jax.random.key(0), **hp)
        model = eqx.tree_deserialise_leaves(f, skeleton)
    return model, hp


def imex_step(carry, _):
    t, y, exp_st, imp_st = carry
    y_star, exp_st = exp_st(advection, t, y, STEP_SIZE, PARAMS)
    y_new, imp_st = imp_st(diffusion, t, y_star, STEP_SIZE, PARAMS)
    return (t + STEP_SIZE, y_new, exp_st, imp_st), None


@jax.jit
def solve(
    y0: jax.Array, exp_st: pdx.AbstractStepper, imp_st: pdx.AbstractStepper
):
    def frame_step(carry, _):
        (t, y, e, i), _ = jax.lax.scan(
            imex_step, carry, length=STEPS_PER_FRAME
        )
        return (t, y, e, i), (t, y)

    _, (t_frames, y_frames) = jax.lax.scan(
        frame_step,
        (jnp.float32(0.0), y0, exp_st, imp_st),
        length=N_FRAMES,
    )
    return t_frames, y_frames


# ---------------------------------------------------------------------------
# Solve and predict
# ---------------------------------------------------------------------------
x = np.linspace(0, L, num=N, endpoint=False, dtype=np.float32)
y0 = jnp.sin(2 * jnp.pi * jnp.asarray(x) / L)

print("Solving Burgers' equation...")
exp_st, imp_st = build_steppers(NU, N, L)
_, y_frames = solve(y0, exp_st, imp_st)
y_truth = np.asarray(y_frames[-1])

print("Running FNO prediction...")
model, hyperparams = load_model(MODEL_PATH)
inp = jnp.stack([jnp.asarray(x), y0], axis=-1)
prediction = np.asarray(model(inp)).squeeze(-1)

rel_err = float(
    jnp.linalg.norm(jnp.array(prediction) - jnp.array(y_truth))
    / jnp.linalg.norm(jnp.array(y_truth))
)
print(f"  Relative L2 error: {rel_err:.4e}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4), layout="tight")

ax.plot(x, np.asarray(y0), color="steelblue", lw=1.5, ls=":", label="$t = 0$")
ax.plot(x, y_truth, color="black", lw=1.8, label="$t = 1$, solver")
ax.plot(
    x,
    prediction,
    color="crimson",
    lw=1.5,
    ls="--",
    alpha=0.85,
    label="$t = 1$, FNO",
)

ax.set_xlabel("$x$")
ax.set_ylabel("$u(x, t)$")
ax.set_title("$u_0 = \\sin(2\\pi x / L)$")
ax.legend(fontsize=10)

out_path = OUTPUT_DIR / "burgers_fno_sine.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved to {out_path}")
