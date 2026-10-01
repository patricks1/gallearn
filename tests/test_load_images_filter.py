import pathlib

import h5py
import pytest

from gallearn import config

TEST_DATA_DIR = pathlib.Path(__file__).parent / 'test_data'
FIREBOX_SNAP = config.config['gallearn_paths']['firebox_snap']
_SNAP_INT = FIREBOX_SNAP.split('_')[0]
CATALOG_PATH = (
    TEST_DATA_DIR
    / f'catalogs_{FIREBOX_SNAP}'
    / f'galaxies_{_SNAP_INT}.hdf5'
)

# Galaxy IDs that have a corresponding image stub in test_data/.
FIXTURE_IDS = {17026, 561}

# Sentinel ID written by make_galaxy_catalog alongside FIXTURE_IDS. No
# image stub exists for it, so it acts as a true negative for the filter.
_NON_FIXTURE_SENTINEL = 99999

# An ID that is absent from the catalog entirely (not a fixture, not the
# sentinel). Used to verify that _extract_galaxy_id results flow into
# the membership check.
_ABSENT_ID = 12345


def _extract_galaxy_id(fname):
    '''Return the integer galaxy ID embedded in an image filename.

    Image filenames follow the pattern
    "object_<id>_host_ugrband_..." or "object_<id>_sate_ugrband_...".
    The ID is the run of digits between the first and second underscores.
    '''
    underscores = [i for i, c in enumerate(fname) if c == '_']
    return int(fname[underscores[0] + 1: underscores[1]])


class TestCatalogFixture:
    def test_fixture_file_exists(self):
        assert CATALOG_PATH.exists(), (
            f'Galaxy catalog fixture not found at {CATALOG_PATH}. '
            f'Run tests/make_ci_data.py to regenerate fixtures.'
        )

    def test_contains_fixture_ids(self):
        with h5py.File(CATALOG_PATH, 'r') as f:
            ids = set(f['galaxyID'][:].astype(int))
        assert FIXTURE_IDS <= ids, (
            f'Fixture IDs {FIXTURE_IDS} missing from catalog. '
            f'Found: {ids}'
        )

    def test_contains_non_fixture_sentinel(self):
        '''Catalog must include the sentinel so filter tests have a true pos.'''
        with h5py.File(CATALOG_PATH, 'r') as f:
            ids = set(f['galaxyID'][:].astype(int))
        assert _NON_FIXTURE_SENTINEL in ids

    def test_absent_id_not_in_catalog(self):
        with h5py.File(CATALOG_PATH, 'r') as f:
            ids = set(f['galaxyID'][:].astype(int))
        assert _ABSENT_ID not in ids


class TestExtractGalaxyId:
    def test_host_filename(self):
        fname = 'object_17026_host_ugrband_FOV12_p600.hdf5'
        assert _extract_galaxy_id(fname) == 17026

    def test_sat_filename(self):
        fname = 'object_561_sate_ugrband_FOV21_p1050.hdf5'
        assert _extract_galaxy_id(fname) == 561

    def test_large_id(self):
        fname = 'object_99999_host_ugrband_FOV10_p500.hdf5'
        assert _extract_galaxy_id(fname) == 99999

    def test_absent_id(self):
        fname = f'object_{_ABSENT_ID}_host_ugrband_FOV10_p500.hdf5'
        assert _extract_galaxy_id(fname) == _ABSENT_ID


class TestInCatalogFilter:
    '''Verify that the catalog ID set and filename extraction combine
    correctly to implement the in_catalog filter from Dataset.load_images.
    '''

    def test_fixture_ids_pass_filter(self):
        '''Both image-stub filenames must pass the in_catalog check.'''
        with h5py.File(CATALOG_PATH, 'r') as f:
            catalog_ids = set(f['galaxyID'][:].astype(int))
        fnames = [
            'object_17026_host_ugrband_FOV12_p600.hdf5',
            'object_561_host_ugrband_FOV21_p1050.hdf5',
        ]
        for fname in fnames:
            assert _extract_galaxy_id(fname) in catalog_ids, (
                f'{fname} should pass the in_catalog filter'
            )

    def test_absent_id_fails_filter(self):
        '''A filename whose ID is not in the catalog must not pass.'''
        with h5py.File(CATALOG_PATH, 'r') as f:
            catalog_ids = set(f['galaxyID'][:].astype(int))
        fname = f'object_{_ABSENT_ID}_host_ugrband_FOV10_p500.hdf5'
        assert _extract_galaxy_id(fname) not in catalog_ids

    def test_filter_mask_shape_and_values(self):
        '''The boolean mask must match the filename list element-for-element.

        One filename uses a fixture ID (passes), one uses the absent ID
        (fails). The mask must be [True, False] in that order.
        '''
        with h5py.File(CATALOG_PATH, 'r') as f:
            catalog_ids = set(f['galaxyID'][:].astype(int))
        fnames = [
            'object_17026_host_ugrband_FOV12_p600.hdf5',
            f'object_{_ABSENT_ID}_host_ugrband_FOV10_p500.hdf5',
        ]
        mask = [_extract_galaxy_id(f) in catalog_ids for f in fnames]
        assert mask == [True, False]
