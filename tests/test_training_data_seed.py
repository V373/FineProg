"""Raw-cache sampling stays comparable despite model/evaluation RNG use."""

import random

import numpy as np
import pytest
import torch

from test_streaming_mamba2 import training_module


def test_data_seed_replays_epoch_sampling_after_unrelated_rng_use(training_module):
    class Samples(torch.utils.data.Dataset):
        def __len__(self):
            return 36

        def __getitem__(self, index):
            steps = np.sort(np.random.choice(np.arange(1, 100 + index), 20, replace=False))
            return index, torch.from_numpy(steps), random.random()

    def epoch_samples(epoch, extra_random_draws):
        torch.rand(extra_random_draws)
        np.random.rand(extra_random_draws)
        for _ in range(extra_random_draws):
            random.random()
        training_module._seed_raw_cache_epoch(42, epoch)
        return list(torch.utils.data.DataLoader(Samples(), batch_size=12, shuffle=True, num_workers=0))

    for epoch in (0, 1, 2000):
        first = epoch_samples(epoch, 7)
        second = epoch_samples(epoch, 101)
        for left, right in zip(first, second):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(epoch_samples(0, 7)[0][1], epoch_samples(1, 7)[0][1])


def test_unset_data_seed_does_not_change_legacy_rng(training_module):
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    training_module._seed_raw_cache_epoch(None, 3)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize("seed", [True, 1.5, -1, 2**32])
def test_invalid_data_seed_rejected(training_module, seed):
    with pytest.raises(ValueError, match="data_seed"):
        training_module._seed_raw_cache_epoch(seed, 0)
