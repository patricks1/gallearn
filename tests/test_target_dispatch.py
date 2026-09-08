'''
Tests for the seam where task and target together decide which
rows a run trains on and how it scales them. The functions live in
gallearn.train; what they dispatch on lives in
gallearn.target_specs.
'''
import numpy as np
import pytest

import torch

from gallearn import target_specs
from gallearn import train


def _vals(values):
    '''Shape raw target values the way load_metadata hands them over,
    (N, 1).'''
    return torch.FloatTensor(values).reshape(-1, 1)


def test_prepare_targets_fits_scaling_on_train_rows_only():
    '''Verify that prepare_targets fits its scaling statistics from
    the rows it is given, never the whole dataset, so no held-out
    galaxy influences the scaling.'''
    raw = _vals([0.1, 0.2, 0.3, 10., 20.])
    d = {'ys_sorted': raw}
    train_idxs = torch.tensor([0, 1, 2])

    _, _, stats = train.prepare_targets(
        'regressor', d, 5, train_idxs, 'fgas'
    )
    expected = target_specs.get('fgas').fit(raw[train_idxs])
    np.testing.assert_allclose(
        stats['means'].numpy(), expected['means'].numpy()
    )
    np.testing.assert_allclose(
        stats['stds'].numpy(), expected['stds'].numpy()
    )


def test_prepare_targets_reuses_cached_stats():
    '''Verify that passing cached statistics skips refitting, so a
    resumed run rescales its targets exactly as the original run did
    rather than refitting against whatever rows it carries.'''
    raw = _vals([0.1, 0.2, 0.3, 0.4])
    d = {'ys_sorted': raw}
    cached = {
        'means': torch.tensor([0.5]),
        'stds': torch.tensor([2.0]),
    }
    targets, _, stats = train.prepare_targets(
        'regressor', d, 4, torch.tensor([0, 1]), 'fgas',
        target_stats=cached,
    )
    assert stats is cached
    np.testing.assert_allclose(
        targets.numpy(), ((raw - 0.5) / 2.0).numpy(), rtol=1e-6
    )


def test_prepare_targets_rejects_classifier_on_fgas():
    '''Verify that asking for the hurdle classifier on a target with
    no structural zero raises, rather than quietly training a
    classifier against a threshold that means nothing.'''
    d = {'ys_sorted': _vals([0., 0.2, 0.4])}
    with pytest.raises(ValueError, match='does not support'):
        train.prepare_targets(
            'classifier', d, 3, torch.tensor([0, 1]), 'fgas'
        )


def test_prepare_targets_classifier_labels_star_forming():
    '''Verify that the classifier labels quenched galaxies 0 and
    star-forming ones 1, and evaluates on every row.'''
    d = {'ys_sorted': _vals([0., 1.e-10, 0.])}
    targets, valid, stats = train.prepare_targets(
        'classifier', d, 3, torch.tensor([0, 1]), 'avg_sfr'
    )
    assert targets.flatten().tolist() == [0., 1., 0.]
    assert valid.tolist() == [0, 1, 2]
    assert stats is None


def test_compute_valid_indices_follows_the_target():
    '''Verify that which rows the regressor trains on comes from the
    target's spec, so a gas-fraction run keeps rows an sSFR run would
    drop.'''
    d = {'ys_sorted': _vals([0., 0.2, 0., 0.4])}
    assert train.compute_valid_indices(
        'regressor', d, 4, 'fgas'
    ).tolist() == [0, 1, 2, 3]
    assert train.compute_valid_indices(
        'regressor', d, 4, 'fdm'
    ).tolist() == [0, 1, 2, 3]
    assert train.compute_valid_indices(
        'regressor', d, 4, 'avg_sfr'
    ).tolist() == [1, 3]
