from collections.abc import Iterator

import numpy as np
from jaxtyping import Array, Shaped


class DataLoader(Iterator):
    """An iterator that yields batches from a dataset."""

    def __init__(
        self,
        dataset: dict[str, Shaped[Array, " n_samples *shape"]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset: Dictionary mapping string keys to arrays.
                All arrays must share the same size along their first axis.
            batch_size: Number of samples per batch. The final batch of an
                epoch may be smaller if the dataset size is not divisible.
            shuffle: Optionally shuffle the samples at the start of each epoch
            seed: Integer seed for the random number generator used to shuffle
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
        self._seed = seed
        self._n_resets = 0
        self.reset()

    def reset(self) -> None:
        """Reset the iterator to the beginning of the dataset."""
        if self.shuffle:
            rng = np.random.default_rng(self._seed)
            self.indices = rng.permutation(self.n_samples)
            self._seed += 1
        else:
            self.indices = np.arange(self.n_samples)
        self.current_idx = 0

    def __next__(self) -> dict[str, Shaped[Array, " batch *shape"]]:
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
