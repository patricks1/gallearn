import pathlib

import numpy as np
import pandas as pd
import pytest

import gallearn.config
import gallearn.dataset_lock
from gallearn import splitting


def test_quantile_edges_collapses_small_n():
    '''Verify that _quantile_edges collapses the requested bin count
    toward the number of available values instead of erroring, and
    still returns real quantile edges when there is enough data.'''
    assert list(splitting._quantile_edges([1., 2., 3.], 10)) == list(
        splitting._quantile_edges([1., 2., 3.], 3)
    )
    assert len(splitting._quantile_edges([1., 2., 3.], 3)) == 2
    assert len(splitting._quantile_edges([1.], 5)) == 0

    edges = splitting._quantile_edges(
        [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], 2
    )
    np.testing.assert_allclose(edges, [5.5])


def test_bin_index():
    '''Verify that _bin_index places values on either side of an edge
    into the correct bin, and a value exactly on an edge goes into
    the higher bin (searchsorted side='right').'''
    edges = np.array([2., 5.])
    assert splitting._bin_index(1., edges) == 0
    assert splitting._bin_index(2., edges) == 1
    assert splitting._bin_index(3., edges) == 1
    assert splitting._bin_index(5., edges) == 2
    assert splitting._bin_index(6., edges) == 2


def test_stratify_galaxies_bins_and_unknown_mass():
    '''Verify that stratify_galaxies routes galaxies missing from
    masses into 'unknown_mass', splits the rest into quenched
    ('ssfr <= 0') vs. star-forming, and never drops a galaxy.'''
    galaxy_ids = ['g1', 'g2', 'g3', 'g4', 'g5']
    masses = {
        'g1': 1.e9, 'g2': 1.e10, 'g3': 1.e11, 'g4': 1.e10,
    }
    ssfrs = {
        'g1': 0., 'g2': 0., 'g3': 1.e-10, 'g4': 1.e-10, 'g5': 1.e-10,
    }

    strata = splitting.stratify_galaxies(
        galaxy_ids, masses, ssfrs, n_mass_bins=2, n_ssfr_bins=2,
    )

    assert strata['unknown_mass'] == ['g5']
    grouped = sorted(gid for ids in strata.values() for gid in ids)
    assert grouped == sorted(galaxy_ids)

    quenched_ids = {
        gid
        for stratum, ids in strata.items()
        if stratum.startswith('quenched_')
        for gid in ids
    }
    assert quenched_ids == {'g1', 'g2'}

    sf_ids = {
        gid
        for stratum, ids in strata.items()
        if stratum.startswith('sf_')
        for gid in ids
    }
    assert sf_ids == {'g3', 'g4'}


def test_stratify_galaxies_small_dataset_degrades():
    '''Verify that stratify_galaxies doesn't error when the galaxy
    count is smaller than the requested bin count, mirroring the CI
    fixture's ~10-galaxy scale against the default 5x5 bins.'''
    galaxy_ids = ['g1', 'g2', 'g3']
    masses = {'g1': 1.e9, 'g2': 1.e10, 'g3': 1.e11}
    ssfrs = {'g1': 1.e-10, 'g2': 1.e-10, 'g3': 1.e-10}

    strata = splitting.stratify_galaxies(
        galaxy_ids, masses, ssfrs, n_mass_bins=5, n_ssfr_bins=5,
    )
    grouped = sorted(gid for ids in strata.values() for gid in ids)
    assert grouped == sorted(galaxy_ids)


def test_select_test_lock_galaxies_never_touches_already_locked():
    '''Verify that select_test_lock_galaxies only draws from galaxies
    outside already_locked, and never returns one of them.'''
    strata = {
        'quenched_m0': ['g1', 'g2', 'g3', 'g4'],
        'sf_m0_s0': ['g5', 'g6', 'g7', 'g8'],
    }
    already_locked = ['g1', 'g5']

    new_galaxies = splitting.select_test_lock_galaxies(
        strata, target_fraction=0.5, already_locked=already_locked,
        seed=42,
    )

    assert not (set(new_galaxies) & set(already_locked))
    assert set(new_galaxies) <= {'g2', 'g3', 'g4', 'g6', 'g7', 'g8'}


def test_select_test_lock_galaxies_skips_unknown_mass():
    '''Verify that select_test_lock_galaxies never selects a galaxy
    out of the 'unknown_mass' stratum, since those galaxies have no
    mass to make the lock representative with.'''
    strata = {
        'unknown_mass': ['g1', 'g2', 'g3', 'g4'],
    }
    new_galaxies = splitting.select_test_lock_galaxies(
        strata, target_fraction=1.0, already_locked=[], seed=42,
    )
    assert new_galaxies == []


def test_select_test_lock_galaxies_reaching_target_is_stable():
    '''Verify that once a stratum's locked share has reached
    target_fraction, a later call with the same target and the
    now-locked galaxies as already_locked selects nothing new,
    matching update_test_lock's no-op-rerun behavior.'''
    strata = {'quenched_m0': ['g1', 'g2', 'g3', 'g4']}
    first = splitting.select_test_lock_galaxies(
        strata, target_fraction=0.5, already_locked=[], seed=42,
    )
    assert len(first) == 2

    second = splitting.select_test_lock_galaxies(
        strata, target_fraction=0.5, already_locked=first, seed=42,
    )
    assert second == []


def test_load_avg_sfr_csv_skips_unparseable_ids(tmp_path):
    '''Verify that load_avg_sfr_csv prepends "object_" to each
    numeric id, parses Mstar as float, and skips (rather than
    crashes on) a row whose id doesn't parse as an integer.'''
    csv_path = tmp_path / 'avg_sfrs.csv'
    csv_path.write_text(
        'id,grp_id,sfr,ssfr,Mstar\n'
        '17026,-1,1.0,1e-10,5.0e9\n'
        '561,-1,0.5,1e-10,2.0e10\n'
        'not_an_id,-1,0.1,1e-10,1.0e10\n'
    )

    masses = splitting.load_avg_sfr_csv(csv_path)

    assert masses == {
        'object_17026': pytest.approx(5.0e9),
        'object_561': pytest.approx(2.0e10),
    }


def test_build_galaxy_index_and_galaxy_ssfr():
    '''Verify that build_galaxy_index groups row indices by galaxy
    and galaxy_ssfr collapses agreeing per-row sSFR copies to a
    single value per galaxy.'''
    obs_sorted = ['object_1', 'object_2', 'object_1']
    ys_sorted = [0.5, 0.0, 0.5]

    galaxy_index = splitting.build_galaxy_index(obs_sorted)
    assert galaxy_index == {'object_1': [0, 2], 'object_2': [1]}

    ssfrs = splitting.galaxy_ssfr(galaxy_index, ys_sorted)
    assert ssfrs == {'object_1': pytest.approx(0.5), 'object_2': 0.0}


def test_galaxy_ssfr_raises_on_disagreement():
    '''Verify that galaxy_ssfr raises rather than silently averaging
    when a galaxy's per-row sSFR copies disagree with each other.'''
    galaxy_index = {'object_1': [0, 1]}
    ys_sorted = [0.5, 0.7]

    with pytest.raises(ValueError):
        splitting.galaxy_ssfr(galaxy_index, ys_sorted)


def test_resolve_split_indices_raises_on_missing_galaxy():
    '''Verify that resolve_split_indices raises ValueError naming a
    split galaxy that's absent from the current dataset's
    galaxy_index, rather than silently dropping it.'''
    galaxy_index = {'object_1': [0], 'object_2': [1]}
    split_dict = {
        'train_galaxies': ['object_1'],
        'val_galaxies': ['object_2', 'object_missing'],
    }

    with pytest.raises(ValueError, match='object_missing'):
        splitting.resolve_split_indices(split_dict, galaxy_index)


def test_resolve_split_indices_returns_row_indices():
    '''Verify that resolve_split_indices maps train/val galaxy ids
    to the correct, sorted row indices from galaxy_index.'''
    galaxy_index = {
        'object_1': [3, 0], 'object_2': [1], 'object_3': [2],
    }
    split_dict = {
        'train_galaxies': ['object_1', 'object_3'],
        'val_galaxies': ['object_2'],
    }

    train_idxs, val_idxs = splitting.resolve_split_indices(
        split_dict, galaxy_index,
    )

    assert train_idxs.tolist() == [0, 2, 3]
    assert val_idxs.tolist() == [1]


@pytest.fixture
def locked_ci_dataset(tmp_path, monkeypatch):
    '''Lock the CI fixture dataset under a tmp_path HASHES_DIR/
    SPLITS_DIR, and point AVG_SFR_CSV at a synthetic per-galaxy mass
    table covering the fixture's 10 galaxy ids. Returns the dataset
    filename, already locked and ready for update_test_lock/
    write_split.'''
    monkeypatch.setattr(splitting, 'SPLITS_DIR', tmp_path / 'splits')
    monkeypatch.setattr(
        gallearn.dataset_lock, 'HASHES_DIR', tmp_path / 'dataset_hashes',
    )

    avg_sfr_csv = tmp_path / 'avg_sfrs.csv'
    avg_sfr_csv.write_text(
        'id,grp_id,sfr,ssfr,Mstar\n'
        '17026,-1,1.0,1e-10,5.0e9\n'
        '561,-1,0.5,1e-10,2.0e10\n'
        '1769,3,0.0,0.0,8.0e10\n'
        '350,-1,2.0,1e-10,1.0e10\n'
        '1420,5,0.0,0.0,3.0e10\n'
        '1406,-1,0.3,1e-10,6.0e9\n'
        '426,-1,0.1,1e-10,9.0e10\n'
        '185,2,0.0,0.0,1.5e10\n'
        '653,-1,0.8,1e-10,4.0e10\n'
        '725,-1,0.2,1e-10,7.0e9\n'
    )
    monkeypatch.setattr(splitting, 'AVG_SFR_CSV', avg_sfr_csv)

    dataset_fname = gallearn.config.config['gallearn_paths']['dataset']
    gallearn.dataset_lock.lock_dataset(dataset_fname)
    return dataset_fname


def test_update_test_lock_top_up_is_append_only(locked_ci_dataset):
    '''Verify that a second update_test_lock call with a higher
    target_fraction only ever adds galaxies on top of the first
    call's locked_galaxies, never removing or replacing any of
    them, and writes a new, higher-versioned file rather than
    editing the first one in place.'''
    dataset_fname = locked_ci_dataset

    first_path, first_lock = splitting.update_test_lock(
        dataset_fname,
        target_fraction=0.2,
        n_mass_bins=2,
        n_ssfr_bins=2,
        avg_sfr_csv=splitting.AVG_SFR_CSV,
    )
    second_path, second_lock = splitting.update_test_lock(
        dataset_fname,
        target_fraction=0.5,
        n_mass_bins=2,
        n_ssfr_bins=2,
        avg_sfr_csv=splitting.AVG_SFR_CSV,
    )

    assert first_lock['lock_version'] == 1
    assert second_lock['lock_version'] == 2
    assert first_path.exists()
    assert second_path.exists()

    first_galaxies = set(first_lock['locked_galaxies'])
    second_galaxies = set(second_lock['locked_galaxies'])
    assert first_galaxies <= second_galaxies

    # The first version's file on disk is untouched by the top-up.
    reloaded_first = splitting._load_json(first_path)
    assert reloaded_first == first_lock


def test_write_split_excludes_locked_galaxies(locked_ci_dataset):
    '''Verify that write_split never places a locked test galaxy
    into train or val.'''
    dataset_fname = locked_ci_dataset

    _, lock_dict = splitting.update_test_lock(
        dataset_fname,
        target_fraction=0.2,
        n_mass_bins=2,
        n_ssfr_bins=2,
        avg_sfr_csv=splitting.AVG_SFR_CSV,
    )
    locked_galaxies = set(lock_dict['locked_galaxies'])
    assert locked_galaxies

    split_dict = splitting.write_split(
        dataset_fname, val_fraction=0.3, split_name='unit',
    )

    split_galaxies = set(
        split_dict['train_galaxies'] + split_dict['val_galaxies']
    )
    assert not (split_galaxies & locked_galaxies)


def test_write_split_refuses_to_overwrite(locked_ci_dataset):
    '''Verify that write_split raises FileExistsError rather than
    silently overwriting a split file already at the target name.'''
    dataset_fname = locked_ci_dataset
    splitting.update_test_lock(
        dataset_fname,
        target_fraction=0.2,
        n_mass_bins=2,
        n_ssfr_bins=2,
        avg_sfr_csv=splitting.AVG_SFR_CSV,
    )
    splitting.write_split(
        dataset_fname, val_fraction=0.3, split_name='dup',
    )

    with pytest.raises(FileExistsError):
        splitting.write_split(
            dataset_fname, val_fraction=0.3, split_name='dup',
        )
