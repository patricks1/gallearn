import pathlib
import queue

import numpy as np
import pandas as pd
import pytest

import gallearn.config
from gallearn import gen_octant_shapes

TEST_DATA = pathlib.Path(__file__).parent / 'test_data'
FIT_COLS = ('b_a', 'PA', 'n', 'Re', 'Ie_log10')

# (galaxyID, view) pairs whose stored astrophot reference never
# converged (its 'converged' status is 'fail. Maximum iterations',
# not 'success'): the LM optimizer hit its iteration cap and reported
# whatever parameter values it landed on at that arbitrary cutoff.
# That landing point is sensitive to floating-point differences
# between machines (BLAS backend, CPU vendor, thread scheduling) even
# with identical package versions, so _check_galaxy excludes exactly
# these known pairs from the reference comparison rather than
# loosening the tolerance for every row.
NONDETERMINISTIC_ASTROPHOT_FITS = frozenset([
    (561, 'projection_ppp'),
])


def _fixture_gal_ids():
    '''Return sorted galaxy IDs discovered from HDF5 files in
    test_data/octant_images/ so the test suite stays valid after
    fixture regeneration without hard-coding specific IDs.

    Returns
    -------
    list of int
        Sorted galaxy IDs found in test_data/octant_images/.
    '''
    return sorted(
        int(p.name.split('_')[1])
        for p in (TEST_DATA / 'octant_images').glob('object_*_*.hdf5')
    )


def _run_galaxy(gal_id, fitter, skip_views=frozenset()):
    '''Call _process_galaxy for a fixture galaxy and collect the row dicts
    sent via queue messages.

    Parameters
    ----------
    gal_id: int
        Galaxy ID; must have exactly one matching HDF5 fixture file.
    skip_views: frozenset of str, default frozenset()
        Projection names to skip (passed through to _process_galaxy).
    fitter: str, default 'fit_sersic'
        Fitting backend passed to _process_galaxy.

    Returns
    -------
    list of dict
        Row dicts in projection order; skipped projections are excluded.
    '''
    matches = sorted(
        (TEST_DATA / 'octant_images').glob(
            f'object_{gal_id}_*_ugrband_*.hdf5'
        )
    )
    assert len(matches) == 1, (
        f'Expected exactly one fixture file for galaxy {gal_id},'
        f' found {matches}'
    )
    hdf5_path = matches[0]
    q = queue.SimpleQueue()
    gen_octant_shapes._process_galaxy(
        (gal_id, str(hdf5_path), q, skip_views, fitter)
    )
    rows = []
    while not q.empty():
        kind, _, payload = q.get()
        if kind == 'row':
            rows.append(payload)
    return rows


def _check_galaxy(gal_id, ref_df, fitter):
    '''Run _process_galaxy for gal_id and assert each fit column matches
    the stored reference within 1% relative tolerance. Skips PA when b_a
    exceeds 0.95 (near-circular; PA is unconstrained), and skips a
    (gal_id, view) pair entirely when it's listed in
    NONDETERMINISTIC_ASTROPHOT_FITS.

    Parameters
    ----------
    gal_id: int
        Galaxy ID to check.
    ref_df: pd.DataFrame
        Reference DataFrame loaded from the reference CSV.
    fitter: str, default 'fit_sersic'
        Fitting backend passed through to _run_galaxy.

    Returns
    -------
    None
    '''
    gal_ref = ref_df[ref_df['galaxyID'] == gal_id].set_index('view')
    rows = _run_galaxy(gal_id, fitter=fitter)
    for row in rows:
        proj = row['view']
        if (gal_id, proj) in NONDETERMINISTIC_ASTROPHOT_FITS:
            continue
        skip_pa = gal_ref.loc[proj, 'b_a'] > 0.95
        for col in FIT_COLS:
            if col == 'PA' and skip_pa:
                continue
            expected = gal_ref.loc[proj, col]
            actual = row[col]
            if np.isnan(expected):
                assert np.isnan(actual), f'{proj} {col}: expected NaN'
            else:
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-2,
                    err_msg=f'{proj} {col} mismatch',
                )


def test_fit_sersic():
    '''Verify that _process_galaxy with fit_sersic returns values matching
    the stored reference for every fixture galaxy. Catches unintended
    changes to the fitting logic or fixture images.'''
    ref_path = TEST_DATA / 'octant_shapes_fitsersic_reference.csv'
    ref_df = pd.read_csv(ref_path)
    for gal_id in _fixture_gal_ids():
        _check_galaxy(gal_id, ref_df, fitter='fit_sersic')


def test_astrophot():
    '''Verify that _process_galaxy with astrophot returns values matching
    the stored AstroPhot reference for every fixture galaxy.'''
    ref_path = TEST_DATA / 'octant_shapes_astrophot_reference.csv'
    ref_df = pd.read_csv(ref_path)
    for gal_id in _fixture_gal_ids():
        _check_galaxy(gal_id, ref_df, fitter='astrophot')


def test_resume_partial_galaxy(tmp_path):
    '''Verify that gen(resume=True) skips already-completed projections
    and fits only the missing ones, without re-fitting or overwriting
    the pre-existing rows. Uses dummy sentinel values so it is
    unambiguous that pre-existing rows were preserved rather than
    re-fitted.'''
    gal_ids = _fixture_gal_ids()
    assert gal_ids, 'No fixture galaxies found in test_data/octant_images'
    test_gal = gal_ids[0]

    dummy_values = {
        'b_a': 0.5, 'PA': 0.1, 'n': 1.0, 'Re': 10.0, 'Ie_log10': 1.0,
    }
    done_views = gen_octant_shapes.OCTANT_PROJECTIONS[:4]
    partial_rows = []
    for view in done_views:
        partial_rows.append({
            'galaxyID': test_gal,
            'FOV': 99.0,
            'pixel': 64,
            'view': view,
            'band': 'band_r',
            **dummy_values,
            # gen(resume=True) defaults to fitter='astrophot', which adds
            # a 'converged' column; this dummy row must already have it so
            # appended rows stay column-aligned with this file's header.
            'converged': 'dummy',
        })
    out_csv = (
        tmp_path / 'AstroPhot_octant_allgals_bandr_Sersic.csv'
    )
    pd.DataFrame(partial_rows).to_csv(out_csv, index=False)

    orig = gallearn.config.config['gallearn_paths']['project_data_dir']
    try:
        gallearn.config.config['gallearn_paths']['project_data_dir'] = (
            str(tmp_path)
        )
        gen_octant_shapes.gen(resume=True)
    finally:
        gallearn.config.config['gallearn_paths']['project_data_dir'] = orig

    result = pd.read_csv(out_csv)
    gal_result = result[result['galaxyID'] == test_gal].copy()

    assert set(gal_result['view']) == set(
        gen_octant_shapes.OCTANT_PROJECTIONS
    ), 'Not all 8 projections present after resume'

    for view in done_views:
        row = gal_result[gal_result['view'] == view].iloc[0]
        assert row['b_a'] == pytest.approx(dummy_values['b_a']), (
            f'{view}: pre-existing b_a was overwritten'
        )
        assert row['Re'] == pytest.approx(dummy_values['Re']), (
            f'{view}: pre-existing Re was overwritten'
        )

    new_views = gen_octant_shapes.OCTANT_PROJECTIONS[4:]
    fitted_views = set(gal_result[~gal_result['view'].isin(done_views)]['view'])
    assert fitted_views == set(new_views), (
        f'Expected new views {new_views}, got {sorted(fitted_views)}'
    )
