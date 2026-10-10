import random
import numpy as np
import torch

from vla.training.train_driving_action_sft import capture_rng_state, restore_rng_state


def test_rng_round_trip():
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
