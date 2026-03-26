import pytest
import jax
import jax.numpy as jnp
import numpy as np

from norax.data import DataLoader


class TestDataLoader:
    key = jax.random.key(0)
    n_samples = 10
    batch_size = 3

    @pytest.fixture
    def dataset(self):
        """Simple dataset with two arrays whose values equal the sample index."""
        x = jnp.arange(self.n_samples, dtype=jnp.float32).reshape(
            self.n_samples, 1
        )
        y = jnp.arange(self.n_samples, dtype=jnp.float32).reshape(
            self.n_samples, 1
        )
        return {"input": x, "output": y}

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DataLoader(self.key, {}, batch_size=4)

    def test_mismatched_sizes_raises(self):
        dataset = {"input": jnp.ones((10, 1)), "output": jnp.ones((8, 1))}
        with pytest.raises(ValueError, match="same number of samples"):
            DataLoader(self.key, dataset, batch_size=4)

    def test_len(self, dataset):
        loader = DataLoader(
            self.key, dataset, batch_size=self.batch_size, shuffle=False
        )
        assert len(loader) == 4

    def test_batch_sizes(self, dataset):
        loader = DataLoader(
            self.key, dataset, batch_size=self.batch_size, shuffle=False
        )
        batches = list(loader)

        for batch in batches[:-1]:
            assert batch["input"].shape[0] == self.batch_size

        last_batch = batches[-1]["input"]
        assert last_batch.shape[0] == self.n_samples % self.batch_size

    def test_all_samples_covered(self, dataset):
        """Every sample index appears exactly once per epoch."""
        loader = DataLoader(
            self.key, dataset, batch_size=self.batch_size, shuffle=False
        )
        seen = np.concatenate([np.array(b["input"]).flatten() for b in loader])
        assert sorted(seen) == list(range(self.n_samples))

    def test_all_keys_present(self, dataset):
        loader = DataLoader(
            self.key, dataset, batch_size=self.batch_size, shuffle=False
        )
        for batch in loader:
            assert set(batch.keys()) == {"input", "output"}

    def test_stop_iteration(self, dataset):
        loader = DataLoader(
            self.key, dataset, batch_size=self.batch_size, shuffle=False
        )
        list(loader)  # exhaust
        with pytest.raises(StopIteration):
            next(loader)

    def test_reset_allows_reiteration(self, dataset):
        loader = DataLoader(
            self.key, dataset, batch_size=self.batch_size, shuffle=False
        )
        first = [np.array(b["input"]) for b in loader]
        loader.reset()
        second = [np.array(b["input"]) for b in loader]
        assert len(first) == len(second)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_no_shuffle_preserves_order(self, dataset):
        loader = DataLoader(
            self.key, dataset, batch_size=self.batch_size, shuffle=False
        )
        seen = np.concatenate([np.array(b["input"]).flatten() for b in loader])
        np.testing.assert_array_equal(
            seen, np.arange(self.n_samples, dtype=np.float32)
        )

    def test_shuffle_covers_all_samples(self, dataset):
        """Shuffled epoch must still cover every sample exactly once."""
        loader = DataLoader(
            self.key, dataset, batch_size=self.batch_size, shuffle=True
        )
        seen = np.concatenate([np.array(b["input"]).flatten() for b in loader])
        assert sorted(seen) == list(range(self.n_samples))

    def test_shuffle_changes_order(self, dataset):
        loader = DataLoader(
            self.key, dataset, batch_size=self.n_samples, shuffle=True
        )

        # Two consecutive epochs should differ
        epoch1 = np.array(next(iter(loader))["input"]).flatten()
        loader.reset()
        epoch2 = np.array(next(iter(loader))["input"]).flatten()
        assert not np.array_equal(epoch1, epoch2)
