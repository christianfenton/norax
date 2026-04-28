import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import install_import_hook

with install_import_hook("norax", "beartype.beartype"):
    from norax.models import FNO, MLP

GRID_CASES = [
    pytest.param((16,), (4,), id="1d"),
    pytest.param((8, 8), (4, 4), id="2d"),
]


def test_mlp_output_shape() -> None:
    key = jax.random.PRNGKey(0)
    model = MLP(key, input_dim=3, output_dim=2, depth=3, width=8)
    x = jax.random.normal(key, (16, 3))
    assert model(x).shape == (16, 2)


class TestFNO:
    @pytest.mark.parametrize("grid_shape,n_modes", GRID_CASES)
    @pytest.mark.parametrize("channels_in,channels_out", [(2, 1), (3, 2)])
    def test_output_shape(
        self,
        grid_shape: tuple[int, ...],
        n_modes: tuple[int, ...],
        channels_in: int,
        channels_out: int,
    ) -> None:
        key = jax.random.PRNGKey(0)
        model = FNO(
            key,
            channels_in=channels_in,
            channels_out=channels_out,
            n_modes=n_modes,
            width=8,
            depth=2,
        )
        x = jax.random.normal(key, (*grid_shape, channels_in))
        assert model(x).shape == (*grid_shape, channels_out)

    @pytest.mark.parametrize("grid_shape,n_modes", GRID_CASES)
    def test_gradient_flow(
        self,
        grid_shape: tuple[int, ...],
        n_modes: tuple[int, ...],
    ) -> None:
        key = jax.random.PRNGKey(0)
        model = FNO(
            key,
            channels_in=2,
            channels_out=1,
            n_modes=n_modes,
            width=8,
            depth=2,
        )
        x = jax.random.normal(key, (*grid_shape, 2))

        def loss(model: FNO, x: jax.Array) -> jax.Array:
            return jnp.mean(model(x) ** 2)

        value, grads = eqx.filter_value_and_grad(loss)(model, x)

        assert jnp.isfinite(value)
        grad_leaves = [g for g in jax.tree.leaves(grads) if eqx.is_array(g)]
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
