import math

import numpy as np
import pytest

from benchtop import stats


def test_shared_context_noise_does_not_shrink_standard_error():
    treatment = dict(enumerate([1.0, 2.0, 3.0, 4.0]))
    baseline = dict.fromkeys(treatment, 0.0)
    delta, se = stats.average_contrast([(treatment, baseline)] * 4)
    assert delta == 2.5
    assert se == pytest.approx(math.sqrt(5 / 3) / 2)


def test_opposing_context_noise_cancels_within_each_seed():
    first = dict(enumerate([1.0, 2.0, 3.0, 4.0]))
    second = {seed: 6 - value for seed, value in first.items()}
    baseline = dict.fromkeys(first, 0.0)
    delta, se = stats.average_contrast([(first, baseline), (second, baseline)])
    assert delta == 3.0
    assert se == 0.0


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [({0: 1.0}, {0: 0.0})],
        [({0: 1.0, 1: 2.0}, {0: 0.0})],
        [({0: 1.0, 1: 2.0}, {0: 0.0, 1: 0.0}), ({0: 1.0}, {0: 0.0})],
        [({0: float("nan"), 1: 2.0}, {0: 0.0, 1: 0.0})],
    ],
)
def test_contrast_requires_complete_finite_seed_blocks(pairs):
    with pytest.raises(ValueError):
        stats.average_contrast(pairs)


def test_small_sample_interval_can_leave_a_normal_label_ambiguous():
    assert stats.label(0.041, 0.01, 0.02, n=12) is None
    assert stats.label(0.001, 0.002, 0.02, n=12) == "no_meaningful_difference"
    assert stats.label(0.02, 0.0, 0.02, n=12) is None


def test_interval_requires_at_least_two_seeds():
    with pytest.raises(ValueError):
        stats.label(0.1, 0.01, 0.02, n=1)


def test_bootstrap_preserves_shared_context_blocks():
    treatment = dict(enumerate(np.linspace(-0.1, 0.1, 12)))
    baseline = dict.fromkeys(treatment, 0.0)
    single = stats.summarize_contrast([(treatment, baseline)], 0.02)
    duplicated = stats.summarize_contrast([(treatment, baseline)] * 4, 0.02)
    assert duplicated == single
    assert single["label"] is None
    assert single["status"] == "ambiguous"
    assert single["n_seeds"] == 12
    assert single["interval"][0] < 0 < single["interval"][1]
