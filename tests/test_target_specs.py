import numpy as np
import pytest

import torch

from gallearn import target_specs


def _vals(values):
    '''Shape raw target values the way load_metadata hands them over,
    (N, 1).'''
    return torch.FloatTensor(values).reshape(-1, 1)


def test_registry_covers_every_target_the_cli_offers():
    '''Verify that every target name is registered and resolves to a
    spec, and that an unknown name raises rather than falling back to
    some default target.'''
    for tgt_type in ['sfr', 'avg_sfr', 'fgas']:
        assert target_specs.get(tgt_type) is not None
    with pytest.raises(ValueError, match='No target spec'):
        target_specs.get('fdm')


def test_sfr_and_avg_sfr_share_one_spec_class():
    '''Verify that both sSFR window choices train identically. They
    differ only in the averaging window the Julia side used, which
    changes the values, not how to handle them.'''
    assert type(target_specs.get('sfr')) is type(
        target_specs.get('avg_sfr')
    )


@pytest.mark.parametrize('tgt_type', ['avg_sfr', 'fgas'])
def test_scale_unscale_round_trips(tgt_type):
    '''Verify that unscale inverts scale for each target, so
    predictions can be mapped back to raw units. This is what lets
    the rest of the pipeline treat the scaling statistics as
    opaque.'''
    spec = target_specs.get(tgt_type)
    if tgt_type == 'fgas':
        raw = _vals([0., 0.02, 0.11, 0.4, 0.9])
    else:
        raw = _vals([1.e-12, 4.e-11, 2.e-10, 9.e-10, 3.e-9])

    stats = spec.fit(raw)
    recovered = spec.unscale(spec.scale(raw, stats), stats)
    np.testing.assert_allclose(
        recovered.numpy(), raw.numpy(), rtol=1e-4, atol=1e-12
    )


def test_ssfr_selects_star_forming_only():
    '''Verify that the sSFR spec trains the regressor on star-forming
    galaxies alone, leaving the structural zeros to the classifier.'''
    spec = target_specs.get('avg_sfr')
    raw = _vals([0., 1.e-10, 0., 5.e-10])
    assert spec.select_valid(raw).tolist() == [1, 3]


def test_fgas_keeps_gas_free_galaxies():
    '''Verify that the gas-fraction spec trains on every galaxy,
    including the fully stripped ones at exactly zero. Those are a
    real physical value rather than a missing-data sentinel, so
    dropping them would discard signal.'''
    spec = target_specs.get('fgas')
    raw = _vals([0., 0.05, 0., 0.3])
    assert spec.select_valid(raw).tolist() == [0, 1, 2, 3]
    assert 'gas-free' in spec.population_summary(raw)


def test_only_ssfr_supports_the_classifier():
    '''Verify that the hurdle classifier is offered for sSFR alone.
    It exists to separate a structural zero from a continuous range,
    which gas fraction does not have.'''
    assert target_specs.get('avg_sfr').supports_classifier
    assert not target_specs.get('fgas').supports_classifier


def test_fgas_avoids_log_axes():
    '''Verify that gas fraction asks for linear axes. A log axis
    silently drops the gas-free galaxies sitting at exactly zero.'''
    assert not target_specs.get('fgas').log_scale
    assert target_specs.get('avg_sfr').log_scale
