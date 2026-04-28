from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .layers import Fourier, Linear


class MLP(eqx.Module):
    "Multi-layer perceptron."

    activation: Callable = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    hidden_layers: tuple
    output_layer: Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int,
        activation: Callable = jax.nn.gelu,
        dtype: jnp.dtype = jnp.result_type(float),
    ) -> None:
        """
        Args:
            key: PRNG key for parameter initialisation
            input_dim: Dimensionality of the input features
            output_dim: Dimensionality of the output
            depth: Number of layers (including output layer)
            width: Number of dimensions in the hidden layers
            activation: Non-linear activation applied between hidden layers.
                Not applied after the final layer.
            dtype: Floating-point dtype of the parameters
        """
        self.activation = activation
        self.width = width
        self.depth = depth

        dims = (input_dim, *((depth - 1) * (width,)), output_dim)

        hidden_layers = []
        for i in range(len(dims) - 2):
            key, subkey = jax.random.split(key)
            hidden_layers.append(
                Linear(subkey, dims[i], dims[i + 1], dtype=dtype)
            )
        self.hidden_layers = tuple(hidden_layers)

        key, subkey = jax.random.split(key)
        self.output_layer = Linear(subkey, dims[-2], dims[-1], dtype=dtype)

    def __call__(
        self, x: Float[Array, "... input_dim"]
    ) -> Float[Array, "... output_dim"]:
        """Perform a forward pass.

        Args:
            x: Input array of shape ``(..., input_dim)``.

        Returns:
            Output array of shape ``(..., output_dim)``.
        """
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output_layer(x)


class FNO(eqx.Module):
    """Fourier neural operator.

    Reference:
        Li et al. "Fourier Neural Operator for Parametric Partial
            Differential Equations" (2020).
    """

    lift: Callable
    project: Callable
    fourier_layers: tuple

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
        lift: Optional[Callable] = None,
        project: Optional[Callable] = None,
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
            lift: Optional custom lifting layer. Must be callable.
                Default: MLP with depth=2 and width=2*width.
            project: Optional custom projection layer. Must be callable.
                Default: MLP with depth=2 and width=2*width.
        """

        # Lifting: (*grid_shape, channels_in) -> (*grid_shape, width)
        if lift is not None:
            self.lift = lift
        else:
            key, subkey = jax.random.split(key)
            self.lift = MLP(
                subkey,
                input_dim=channels_in,
                output_dim=width,
                depth=2,
                width=2 * width,
                activation=activation,
                dtype=dtype,
            )

        # Fourier layers: (*grid_shape, width) -> (*grid_shape, width)
        fourier_layers = []
        for _ in range(depth):
            key, subkey = jax.random.split(key)
            fourier_layers.append(
                Fourier(
                    subkey,
                    channels_in=width,
                    channels_out=width,
                    n_modes=n_modes,
                    activation=activation,
                    dtype=dtype,
                )
            )
        self.fourier_layers = tuple(fourier_layers)

        # Projection: (*grid_shape, width) -> (*grid_shape, channels_out)
        if project is not None:
            self.project = project
        else:
            key, subkey = jax.random.split(key)
            self.project = MLP(
                subkey,
                input_dim=width,
                output_dim=channels_out,
                depth=2,
                width=2 * width,
                activation=activation,
                dtype=dtype,
            )

    def __call__(
        self, x: Float[Array, "*grid_shape channels_in"]
    ) -> Float[Array, "*grid_shape channels_out"]:
        """Perform a forward pass.

        Args:
            x: Input array with shape ``(*grid_shape, channels_in)``
                For example, a 1D input with a single channel would have shape
                ``(n, 1)``, while a 2D input with a single channel
                would have shape ``(nx, ny, 1)``.

        Returns:
            Output array with shape ``(*grid_shape, channels_out)``.
        """
        x = self.lift(x)

        for layer in self.fourier_layers:
            x = layer(x)

        return self.project(x)
