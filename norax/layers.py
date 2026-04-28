from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .initialisers import complex_glorot


class Linear(eqx.Module):
    """A pointwise linear transformation applied along the last axis."""

    weight: Float[Array, "output_dim input_dim"]
    bias: Float[Array, " output_dim"]

    def __init__(
        self,
        key: PRNGKeyArray,
        input_dim: int,
        output_dim: int,
        init: Callable = jax.nn.initializers.glorot_uniform(),
        dtype: jnp.dtype = jnp.result_type(float),
    ) -> None:
        """
        Args:
            key: PRNG key for parameter initialisation
            input_dim: Number of input channels
            output_dim: Number of output channels
            init: Weight initialiser
            dtype: Floating-point dtype of the parameters
        """
        self.weight = init(key, (output_dim, input_dim), dtype=dtype)
        self.bias = jnp.zeros(output_dim, dtype=dtype)

    def __call__(
        self, x: Float[Array, "... input_dim"]
    ) -> Float[Array, "... output_dim"]:
        """Apply a linear transformation along the last axis.

        Args:
            x: Array with shape (..., channels_in)

        Returns:
            Array with shape (..., channels_out)
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
    n_dims: int = eqx.field(static=True)
    weights_re: Float[Array, "channels_out channels_in *weight_modes"]
    weights_im: Float[Array, "channels_out channels_in *weight_modes"]

    def __init__(
        self,
        key: PRNGKeyArray,
        channels_in: int,
        channels_out: int,
        n_modes: tuple[int, ...],
        init: Callable = complex_glorot,
        dtype: jnp.dtype = jnp.result_type(float),
    ) -> None:
        """
        Args:
            key: PRNG key for parameter initialisation
            channels_in: Number of input channels
            channels_out: Number of output channels
            n_modes: Number of modes to retain per axis. For non-last axes,
                ``n_modes[i]`` modes are kept at each end of the spectrum
                (positive and negative), so the stored weight size is
                ``2 * n_modes[i]``. For the last axis the one-sided RFFT
                spectrum is used, so ``n_modes[-1]`` weights are stored.
            init: Complex weight initialiser used to draw the initial
                real and imaginary components
            dtype: Floating-point dtype of the parameters
        """
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_modes = n_modes
        self.n_dims = len(n_modes)

        mode_shape = tuple(
            2 * m if i < len(n_modes) - 1 else m for i, m in enumerate(n_modes)
        )
        sh = (channels_out, channels_in) + mode_shape
        w = init(key, shape=sh, dtype=dtype)
        self.weights_re = w.real
        self.weights_im = w.imag

    def __call__(
        self, x: Float[Array, "*coords channels_in"]
    ) -> Float[Array, "*coords channels_out"]:
        """Perform a spectral convolution using an FFT.

        Args:
            x: Input array with shape (*coords, channels_in)

        Returns:
            Array with shape (*coords, channels_out)
        """
        coords = x.shape[: self.n_dims]
        coord_axes = tuple(range(self.n_dims))

        # Transform to spectral space
        Fx = jnp.fft.rfftn(x, s=coords, axes=coord_axes, norm="ortho")
        rfft_shape = Fx.shape

        # Truncate to retained modes
        slices = []
        for i in range(self.n_dims):
            n_freq = Fx.shape[i]
            k = self.n_modes[i]

            is_last = i == self.n_dims - 1
            limit = n_freq if is_last else n_freq // 2
            if k > limit:
                raise ValueError(
                    f"n_modes[{i}] = {k} exceeds the maximum of {limit} "
                    f"for axis {i} (resolution {n_freq}). "
                    f"Reduce n_modes or increase the spatial resolution."
                )

            if i < self.n_dims - 1:
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
        weights = self.weights_re + 1j * self.weights_im
        _DIM_LABELS = "abcdefghjklmnpqrstuvwxyz"  # excludes 'i' and 'o'
        mode_labels = _DIM_LABELS[: self.n_dims]
        ein_str = f"oi{mode_labels},{mode_labels}i->{mode_labels}o"
        Fv_trunc = jnp.einsum(ein_str, weights, Fx_trunc)

        # Place the truncated modes into a full-sized array
        sh = rfft_shape[: self.n_dims]
        Fv = jnp.zeros((*sh, self.channels_out), dtype=Fv_trunc.dtype)
        Fv = Fv.at[jnp.ix_(*slices)].set(Fv_trunc)

        # Transform back to physical space
        out = jnp.fft.irfftn(Fv, s=coords, axes=coord_axes, norm="ortho")

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
        key: PRNGKeyArray,
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
            n_modes: Tuple of maximum number of modes per coordinate axis
            activation: Non-linear activation function
            dtype: Floating-point dtype of the parameters
        """
        key1, key2 = jax.random.split(key)
        self.spectral = SpectralConv(
            key1, channels_in, channels_out, n_modes, dtype=dtype
        )
        self.linear = Linear(key2, channels_in, channels_out, dtype=dtype)
        self.activation = activation

    def __call__(
        self, x: Float[Array, "*coords channels_in"]
    ) -> Float[Array, "*coords channels_out"]:
        """Apply the Fourier layer.

        The layer computes v_{t+1} = sigma(W v_t + F^{-1}[R * F(v_t)]),
        where W is a trainable pointwise linear transformation,
        F is the fast Fourier transform,
        R contains trainable complex weights for the retained Fourier modes,
        sigma is a non-linear activation function,
        and * denotes element-wise multiplication.

        Args:
            x: Input tensor of shape (*coords, channels_in)

        Returns:
            Array with shape (*coords, channels_out)
        """
        return self.activation(self.spectral(x) + self.linear(x))
