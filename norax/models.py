from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .layers import Fourier, Linear


class MLP(eqx.Module):
    "Multi-layer perceptron."

    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    layers: list
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        sizes: tuple[int, ...],
        activation: Callable = jax.nn.gelu,
        dtype: jnp.dtype = jnp.result_type(float),
    ) -> None:
        """
        Args:
            key: PRNG key for parameter initialisation
            sizes: Tuple of layer widths ``(input_dim, hidden..., output_dim)``
            activation: Non-linear activation applied between hidden layers.
                Not applied after the final layer.
            dtype: Floating-point dtype of the parameters
        """
        if len(sizes) < 2:
            raise ValueError(
                "sizes must contain at least two elements (input and output)."
            )

        self.input_dim = sizes[0]
        self.output_dim = sizes[-1]
        self.depth = len(sizes) - 1
        self.width = max(sizes[1:-1]) if len(sizes) > 2 else 0
        self.activation = activation

        layers = []
        for i in range(len(sizes) - 1):
            key, subkey = jax.random.split(key)
            layers.append(Linear(subkey, sizes[i], sizes[i + 1], dtype=dtype))
        self.layers = layers

    def __call__(
        self, x: Float[Array, "*batch input_dim"]
    ) -> Float[Array, "*batch output_dim"]:
        """Perform a forward pass.

        Args:
            x: Input array of shape ``(*batch_dims, input_dim)``

        Returns:
            Output array of shape ``(*batch_dims, output_dim)``
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class FNO(eqx.Module):
    """
    Fourier neural operator.

    The input is expected to have shape (*coords, channels_in).

    Reference:
        Li et al. "Fourier Neural Operator for Parametric Partial
            Differential Equations" (2020).
    """

    depth: int = eqx.field(static=True)

    lift: eqx.Module
    fourier_layers: list
    project: eqx.Module

    def __init__(
        self,
        key: PRNGKeyArray,
        channels_in: int,
        channels_out: int,
        n_modes: tuple[int, ...],
        width: int = 64,
        depth: int = 4,
        activation: Callable = jax.nn.gelu,
        dtype: jnp.dtype = jnp.result_type(float),
        lift: Optional[eqx.Module] = None,
        project: Optional[eqx.Module] = None,
    ) -> None:
        """
        Args:
            key: PRNG key for parameter initialisation
            channels_in: Number of input channels
            channels_out: Number of output channels
            n_modes: Tuple of maximum number of modes per coordinate axis
            width: Hidden channel width
            depth: Number of Fourier layers
            activation: Non-linear activation function
            dtype: Floating-point dtype of the parameters
            lift: Optional custom lifting module
            project: Optional custom projection module
        """
        self.depth = depth

        # Lifting: (*coords, channels_in) -> (*coords, width)
        if lift is not None:
            self.lift = lift
        else:
            key, subkey = jax.random.split(key)
            self.lift = MLP(subkey, (channels_in, width), dtype=dtype)

        # Fourier layers: width -> width
        fourier_layers = []
        for _ in range(depth):
            key, subkey = jax.random.split(key)
            layer = Fourier(
                subkey, width, width, n_modes, activation, dtype=dtype
            )
            fourier_layers.append(layer)
        self.fourier_layers = fourier_layers

        # Projection: (*coords, width) -> (*coords, channels_out)
        if project is not None:
            self.project = project
        else:
            key, subkey = jax.random.split(key)
            self.project = MLP(
                subkey, (width, width, channels_out), activation, dtype=dtype
            )

    def __call__(
        self, x: Float[Array, "*coords channels_in"]
    ) -> Float[Array, "*coords channels_out"]:
        """Perform a forward pass.

        Args:
            x: Input tensor of shape (*coords, channels_in)

        Returns:
            Output tensor of shape (*coords, channels_out)
        """
        x = self.lift(x)

        for layer in self.fourier_layers:
            x = layer(x)

        x = self.project(x)

        return x
