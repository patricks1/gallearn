import json
import pathlib
import queue

import numpy as np

from gallearn import gen_octant_shapes

TEST_DATA = pathlib.Path(__file__).parent / 'test_data'
FIT_COLS = ('b_a', 'PA', 'n', 'Re', 'Ie')
FIXTURE_GAL_IDS = (112, 910)


def _check_galaxy(gal_id, ref):
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
    gal_ref = ref[str(gal_id)]
    rows = gen_octant_shapes._process_galaxy(
        (gal_id, str(hdf5_path), queue.SimpleQueue())
    )
    for row in rows:
        proj = row['view']
        # When b_a is near 1 the galaxy is nearly circular and PA is
        # unconstrained -- the optimizer can return any angle, so skip it.
        skip_pa = gal_ref[proj]['b_a'] > 0.95
        for col in FIT_COLS:
            if col == 'PA' and skip_pa:
                continue
            expected = gal_ref[proj][col]
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


def test_process_galaxy_regression():
    ref_path = TEST_DATA / 'octant_shapes_reference.json'
    with open(ref_path) as fh:
        ref = json.load(fh)
    for gal_id in FIXTURE_GAL_IDS:
        _check_galaxy(gal_id, ref)
