import jax
import jax.numpy as jnp
import pytest
from jaxtyping import install_import_hook

with install_import_hook("norax", "beartype.beartype"):
    from norax.layers import Fourier, Linear, SpectralConv

GRID_CASES = [
    pytest.param((16,), (4,), id="1d"),
    pytest.param((8, 8), (4, 4), id="2d"),
]


def test_linear_output_shape() -> None:
    key = jax.random.PRNGKey(0)
    layer = Linear(key, input_dim=3, output_dim=5)
    x = jax.random.normal(key, (4, 16, 3))
    assert layer(x).shape == (4, 16, 5)


class TestSpectralConv:
    @pytest.mark.parametrize("grid_shape,n_modes", GRID_CASES)
    @pytest.mark.parametrize("channels_in,channels_out", [(2, 4), (4, 1)])
    def test_output_shape(
        self,
        grid_shape: tuple[int, ...],
        n_modes: tuple[int, ...],
        channels_in: int,
        channels_out: int,
    ) -> None:
        key = jax.random.PRNGKey(0)
        layer = SpectralConv(key, channels_in, channels_out, n_modes)
        x = jax.random.normal(key, (*grid_shape, channels_in))
        assert layer(x).shape == (*grid_shape, channels_out)

    def test_weights_stored_as_real(self) -> None:
        key = jax.random.PRNGKey(0)
        layer = SpectralConv(key, channels_in=2, channels_out=4, n_modes=(4,))
        assert jnp.isrealobj(layer.weights_re)
        assert jnp.isrealobj(layer.weights_im)


class TestFourier:
    @pytest.mark.parametrize("grid_shape,n_modes", GRID_CASES)
    def test_output_shape(
        self,
        grid_shape: tuple[int, ...],
        n_modes: tuple[int, ...],
    ) -> None:
        key = jax.random.PRNGKey(0)
        layer = Fourier(key, channels_in=4, channels_out=8, n_modes=n_modes)
        x = jax.random.normal(key, (*grid_shape, 4))
        assert layer(x).shape == (*grid_shape, 8)
