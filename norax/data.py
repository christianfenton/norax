from collections.abc import Iterator
from typing import Dict

import jax
import jax.numpy as jnp


class DataLoader(Iterator):
    """An iterator that yields batches from a dataset."""

    def __init__(
        self,
        key,
        dataset: Dict[str, jax.Array],
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        Args:
            key: PRNG key used to shuffle the dataset
            dataset: Dictionary mapping string keys to arrays.
                All arrays must share the same size along their first axis.
            batch_size: Number of samples per batch. The final batch of an
                epoch may be smaller if the dataset size is not divisible.
            shuffle: Optionally shuffle the samples at the start of each epoch
        """
        if not dataset:
            raise ValueError("dataset must not be empty.")

        sizes = {k: v.shape[0] for k, v in dataset.items()}
        if len(set(sizes.values())) > 1:
            raise ValueError(
                f"All arrays must have the same number of samples. Got: {sizes}"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = next(iter(dataset.values())).shape[0]
        self._key = key
        self.reset()

    def reset(self) -> None:
        """Reset the iterator to the beginning of the dataset."""
        self.indices = jnp.arange(self.n_samples)
        if self.shuffle:
            self._key, subkey = jax.random.split(self._key)
            self.indices = jax.random.permutation(subkey, self.indices)
        self.current_idx = 0

    def __next__(self) -> Dict[str, jax.Array]:
        """Return the next batch.

        Returns:
            Dictionary with the same keys as the dataset

        Raises:
            StopIteration: When all samples have been yielded
        """
        if self.current_idx >= self.n_samples:
            raise StopIteration

        start = self.current_idx
        end = min(start + self.batch_size, self.n_samples)
        batch_indices = self.indices[start:end]

        self.current_idx = end
        return {k: v[batch_indices] for k, v in self.dataset.items()}

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size
