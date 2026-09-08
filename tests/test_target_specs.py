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
    for tgt_type in ['sfr', 'avg_sfr', 'fgas', 'fdm']:
        assert target_specs.get(tgt_type) is not None
    with pytest.raises(ValueError, match='No target spec'):
        target_specs.get('unknown_target')


def test_sfr_and_avg_sfr_share_one_spec_class():
    '''Verify that both sSFR window choices train identically. They
    differ only in the averaging window the Julia side used, which
    changes the values, not how to handle them.'''
    assert type(target_specs.get('sfr')) is type(
        target_specs.get('avg_sfr')
    )


@pytest.mark.parametrize('tgt_type', ['avg_sfr', 'fgas', 'fdm'])
def test_scale_unscale_round_trips(tgt_type):
    '''Verify that unscale inverts scale for each target, so
    predictions can be mapped back to raw units. This is what lets
    the rest of the pipeline treat the scaling statistics as
    opaque.'''
    spec = target_specs.get(tgt_type)
    if tgt_type == 'fgas':
        raw = _vals([0., 0.02, 0.11, 0.4, 0.9])
    elif tgt_type == 'fdm':
        raw = _vals([1.e-6, 0.1, 0.5, 0.8, 0.999])
    else:
        raw = _vals([1.e-12, 4.e-11, 2.e-10, 9.e-10, 3.e-9])

    stats = spec.fit(raw)
    recovered = spec.unscale(spec.scale(raw, stats), stats)
    # atol=1e-12 works for avg_sfr's tiny-but-well-scaled raw values
    # (asinh(stretch * x) lands near O(0.1) before standardizing) and
    # for fgas's exact 0. fdm's near-zero 1e-6, run through plain
    # standardization instead of asinh, keeps float32 rounding noise
    # around 1e-8 in absolute terms, large enough relative to 1e-6
    # to trip rtol=1e-4 without a looser atol.
    atol = 1e-6 if tgt_type == 'fdm' else 1e-12
    np.testing.assert_allclose(
        recovered.numpy(), raw.numpy(), rtol=1e-4, atol=atol
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


def test_fdm_trains_on_every_galaxy():
    '''Verify that the dark-matter-fraction spec trains on every
    galaxy. Unlike gas fraction, no galaxy sits at exactly zero or
    one, so there is no edge case to carve out at all.'''
    spec = target_specs.get('fdm')
    raw = _vals([1.e-6, 0.4, 0.8, 0.999])
    assert spec.select_valid(raw).tolist() == [0, 1, 2, 3]


def test_only_ssfr_supports_the_classifier():
    '''Verify that the hurdle classifier is offered for sSFR alone.
    It exists to separate a structural zero from a continuous range,
    which neither fraction target has.'''
    assert target_specs.get('avg_sfr').supports_classifier
    assert not target_specs.get('fgas').supports_classifier
    assert not target_specs.get('fdm').supports_classifier


def test_fgas_avoids_log_axes():
    '''Verify that gas fraction asks for linear axes. A log axis
    silently drops the gas-free galaxies sitting at exactly zero.'''
    assert not target_specs.get('fgas').log_scale
    assert target_specs.get('avg_sfr').log_scale


def test_fdm_avoids_log_axes():
    '''Verify that dark-matter fraction also asks for linear axes,
    matching gas fraction's scaling choice.'''
    assert not target_specs.get('fdm').log_scale
