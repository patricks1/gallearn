import json
import pathlib
import queue

import numpy as np

from gallearn import gen_octant_shapes

TEST_DATA = pathlib.Path(__file__).parent / 'test_data'
FIT_COLS = ('b_a', 'PA', 'n', 'Re', 'Ie')


def test_process_galaxy_regression():
    matches = sorted(
        (TEST_DATA / 'octant_images').glob('object_112_*_ugrband_*.hdf5')
    )
    assert len(matches) == 1, (
        f'Expected exactly one fixture file for galaxy 112, found {matches}'
    )
    hdf5_path = matches[0]
    ref_path = TEST_DATA / 'octant_shapes_reference.json'
    rows = gen_octant_shapes._process_galaxy(
        (112, str(hdf5_path), queue.SimpleQueue())
    )
    with open(ref_path) as fh:
        ref = json.load(fh)
    for row in rows:
        proj = row['view']
        for col in FIT_COLS:
            expected = ref[proj][col]
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
