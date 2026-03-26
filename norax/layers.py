from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from .initialisers import complex_glorot


class Linear(eqx.Module):
    """A pointwise linear transformation applied along the last axis."""

    weight: jax.Array
    bias: jax.Array

    def __init__(
        self,
        key,
        channels_in: int,
        channels_out: int,
        init: Callable = jax.nn.initializers.glorot_uniform(),
        dtype: jnp.dtype = jnp.result_type(float)
    ) -> None:
        """
        Args:
            key: PRNG key for parameter initialisation
            channels_in: Number of input channels
            channels_out: Number of output channels
            init: Weight initialiser
            dtype: Floating-point dtype of the parameters
        """
        self.weight = init(key, (channels_out, channels_in), dtype=dtype)
        self.bias = jnp.zeros(channels_out, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply a linear transformation along the last axis.

        Args:
            x: Array with shape (*spatial_dims, channels_in)

        Returns:
            Array with shape (*spatial_dims, channels_out)
        """
        return jnp.einsum("...i,ji->...j", x, self.weight) + self.bias


class SpectralConv(eqx.Module):
    """A spectral convolution layer used in the Fourier neural operator.

    Reference:
        Li et al. "Fourier Neural Operator for Parametric Partial
            Differential Equations" (2020).
    """
    channels_in: int = eqx.field(static=True)
    channels_out: int = eqx.field(static=True)
    n_modes: tuple[int, ...] = eqx.field(static=True)
    n_spatial_dims: int = eqx.field(static=True)
    weights: jax.Array

    def __init__(
        self,
        key,
        channels_in: int,
        channels_out: int,
        n_modes: tuple[int, ...],
        init: Callable = complex_glorot,
        dtype = jnp.result_type(float)
    ) -> None:
        """
        Args:
            key: PRNG key for parameter initialisation
            channels_in: Number of input channels
            channels_out: Number of output channels
            n_modes: Maximum number of Fourier modes to retain per spatial axis
            init: Complex weight initialiser
            dtype: Floating-point dtype for the real components of the weights
        """
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_modes = n_modes
        self.n_spatial_dims = len(n_modes)
        
        weight_shape = (channels_out, channels_in) + tuple(n_modes)
        self.weights = init(key, shape=weight_shape, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Perform a spectral convolution using an FFT.

        Args:
            x: Input array with shape (*spatial_dims, channels_in)

        Returns:
            Array with shape (*spatial_dims, channels_out)
        """
        spatial_dims = x.shape[:self.n_spatial_dims]
        spatial_axes = tuple(range(self.n_spatial_dims))

        # Transform to spectral space
        Fx = jnp.fft.rfftn(x, s=spatial_dims, axes=spatial_axes, norm="ortho")
        rfft_shape = Fx.shape

        # Truncate to retained modes
        slices = []
        for i in range(self.n_spatial_dims):
            n_freq = Fx.shape[i]
            k = self.n_modes[i]

            if k > n_freq:
                raise ValueError(
                    f"n_modes[{i}] = {k} exceeds the number of "
                    f"available frequencies ({n_freq}) along axis {i}. "
                    f"Increase the spatial resolution or reduce n_modes."
                )

            if i < self.n_spatial_dims - 1:
                pos = list(range(k))
                neg = list(range(Fx.shape[i] - k, Fx.shape[i]))
                slices.append(jnp.array(pos + neg))
            else:
                # Last axis has half the number of points after RFFT
                slices.append(jnp.array(list(range(k))))

        Fx_trunc = Fx[jnp.ix_(*slices)]

        # Contract over input channels
        #   weights:  (channels_out, channels_in, mode_0, mode_1, ...)
        #   Fx_trunc: (mode_0, mode_1, ..., channels_in)
        #   result:   (mode_0, mode_1, ..., channels_out)
        #
        # 2D example: "oiab,abi->abo"
        _SPATIAL_LABELS = "abcdefghjklmnpqrstuvwxyz"  # excludes 'i' and 'o'
        mode_labels = _SPATIAL_LABELS[:self.n_spatial_dims]
        ein_str = f"oi{mode_labels},{mode_labels}i->{mode_labels}o"
        Fv_trunc = jnp.einsum(ein_str, self.weights, Fx_trunc)

        # Place the truncated modes into a full-sized array
        sh = rfft_shape[:self.n_spatial_dims]
        Fv = jnp.zeros((*sh, self.channels_out), dtype=Fv_trunc.dtype)
        Fv = Fv.at[jnp.ix_(*slices)].set(Fv_trunc)

        # Transform back to physical space
        out = jnp.fft.irfftn(Fv, s=spatial_dims, axes=spatial_axes, norm="ortho")

        return out
    

class Fourier(eqx.Module):
    """A Fourier layer for a Fourier neural operator.

    Reference:
        Li et al. "Fourier Neural Operator for Parametric Partial
            Differential Equations" (2020).
    """
    spectral: SpectralConv
    linear: Linear
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        key,
        channels_in: int,
        channels_out: int,
        n_modes: tuple[int, ...],
        activation: Callable = jax.nn.gelu,
        dtype: jnp.dtype = jnp.result_type(float),
    ) -> None:
        """
        Args:
            key: PRNG key for parameter initialisation
            channels_in: Number of input channels
            channels_out: Number of output channels
            n_modes: Tuple of maximum Fourier modes per spatial axis
            activation: Non-linear activation function
            dtype: Floating-point dtype of the parameters
        """
        key1, key2 = jax.random.split(key)
        self.spectral = SpectralConv(key1, channels_in, channels_out, n_modes, dtype=dtype)
        self.linear = Linear(key2, channels_in, channels_out, dtype=dtype)
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the Fourier layer.

        The layer computes v_{t+1} = sigma(W v_t + F^{-1}[R * F(v_t)]),
        where W is a trainable pointwise linear transformation, 
        F is the fast Fourier transform, 
        R contains trainable complex weights for the retained Fourier modes,
        sigma is a non-linear activation function,
        and * denotes element-wise multiplication.

        Args:
            x: Input tensor of shape (*spatial_dims, channels_in)

        Returns:
            Array with shape (*spatial_dims, channels_out)
        """
        return self.activation(self.spectral(x) + self.linear(x))