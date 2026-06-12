import pathlib

import h5py
import numpy as np
import pytest

import gallearn.gen_octant_images


TEST_DATA_DIR = pathlib.Path(__file__).parent / 'test_data'


# ── rotation_matrix_to_z ──────────────────────────────────────────


class TestRotationMatrixToZ:

    @pytest.mark.parametrize(
        'label,n', list(gallearn.gen_octant_images.OCTANT_DIRECTIONS.items()),
    )
    def test_maps_n_to_z(self, label, n):
        R = gallearn.gen_octant_images.rotation_matrix_to_z(n)
        z_hat = np.array([0., 0., 1.])
        np.testing.assert_allclose(
            R @ n, z_hat, atol=1e-12,
        )

    @pytest.mark.parametrize(
        'label,n', list(gallearn.gen_octant_images.OCTANT_DIRECTIONS.items()),
    )
    def test_orthogonal(self, label, n):
        R = gallearn.gen_octant_images.rotation_matrix_to_z(n)
        np.testing.assert_allclose(
            R @ R.T, np.eye(3), atol=1e-12,
        )

    @pytest.mark.parametrize(
        'label,n', list(gallearn.gen_octant_images.OCTANT_DIRECTIONS.items()),
    )
    def test_proper_rotation(self, label, n):
        R = gallearn.gen_octant_images.rotation_matrix_to_z(n)
        assert np.linalg.det(R) == pytest.approx(
            1.0, abs=1e-12,
        )

    def test_identity_for_z_hat(self):
        R = gallearn.gen_octant_images.rotation_matrix_to_z([0., 0., 1.])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_antiparallel_z(self):
        R = gallearn.gen_octant_images.rotation_matrix_to_z([0., 0., -1.])
        z_hat = np.array([0., 0., 1.])
        np.testing.assert_allclose(
            R @ np.array([0., 0., -1.]),
            z_hat,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            R @ R.T, np.eye(3), atol=1e-12,
        )

    def test_unnormalized_input(self):
        n = np.array([3., 4., 0.])
        R = gallearn.gen_octant_images.rotation_matrix_to_z(n)
        z_hat = np.array([0., 0., 1.])
        n_hat = n / np.linalg.norm(n)
        np.testing.assert_allclose(
            R @ n_hat, z_hat, atol=1e-12,
        )


# ── rotate_snapdict ───────────────────────────────────────────────


class TestRotateSnapdict:

    def test_coordinates_rotated(self):
        coords = np.array([[1., 0., 0.], [0., 1., 0.]])
        snapdict = {
            'Coordinates': coords,
            'r': np.array([1., 1.]),
            'other_key': 'preserve_me',
        }
        R = gallearn.gen_octant_images.rotation_matrix_to_z([1., 0., 0.])
        result = gallearn.gen_octant_images.rotate_snapdict(snapdict, R)
        # x-axis mapped to z-axis, so [1,0,0] -> [0,0,1]
        np.testing.assert_allclose(
            result['Coordinates'][0],
            [0., 0., 1.],
            atol=1e-12,
        )

    def test_other_keys_preserved(self):
        snapdict = {
            'Coordinates': np.array([[1., 0., 0.]]),
            'r': np.array([1.]),
            'Masses': np.array([42.]),
        }
        R = np.eye(3)
        result = gallearn.gen_octant_images.rotate_snapdict(snapdict, R)
        assert result['Masses'] is snapdict['Masses']
        assert result['r'] is snapdict['r']

    def test_original_unchanged(self):
        coords = np.array([[1., 0., 0.]])
        snapdict = {
            'Coordinates': coords,
            'r': np.array([1.]),
        }
        R = gallearn.gen_octant_images.rotation_matrix_to_z([1., 0., 0.])
        gallearn.gen_octant_images.rotate_snapdict(snapdict, R)
        np.testing.assert_array_equal(
            snapdict['Coordinates'], coords,
        )


# ── scan_image_dirs ───────────────────────────────────────────────


class TestScanImageDirs:

    def test_finds_galaxies(self):
        host_dir = TEST_DATA_DIR / 'host_band_ugr'
        sat_dir = TEST_DATA_DIR / 'sat_band_ugr'
        galaxies = gallearn.gen_octant_images.scan_image_dirs(host_dir, sat_dir)
        ids = {g['attrs']['galaxyID'] for g in galaxies}
        assert 768 in ids or 768.0 in ids
        assert 1271 in ids or 1271.0 in ids

    def test_reads_fov_and_pixels(self):
        host_dir = TEST_DATA_DIR / 'host_band_ugr'
        sat_dir = TEST_DATA_DIR / 'sat_band_ugr'
        galaxies = gallearn.gen_octant_images.scan_image_dirs(host_dir, sat_dir)
        by_id = {
            int(g['attrs']['galaxyID']): g
            for g in galaxies
        }
        assert by_id[768]['fov'] == 15
        assert by_id[768]['pixels'] == 750
        assert by_id[1271]['fov'] == 13
        assert by_id[1271]['pixels'] == 650

    def test_deduplicates_by_galaxy_id(self):
        # Scanning the same directory twice should not
        # produce duplicates.
        host_dir = TEST_DATA_DIR / 'host_band_ugr'
        galaxies = gallearn.gen_octant_images.scan_image_dirs(host_dir, host_dir)
        ids = [
            int(g['attrs']['galaxyID']) for g in galaxies
        ]
        assert len(ids) == len(set(ids))

    def test_missing_dir_warns(self, capsys):
        galaxies = gallearn.gen_octant_images.scan_image_dirs(
            '/nonexistent/path', '/also/nonexistent',
        )
        assert galaxies == []
        captured = capsys.readouterr()
        assert 'Warning' in captured.out

    def test_attrs_copied(self):
        host_dir = TEST_DATA_DIR / 'host_band_ugr'
        sat_dir = TEST_DATA_DIR / 'sat_band_ugr'
        galaxies = gallearn.gen_octant_images.scan_image_dirs(host_dir, sat_dir)
        for gal in galaxies:
            attrs = gal['attrs']
            assert 'FOV' in attrs
            assert 'pixels' in attrs
            assert 'galaxyID' in attrs
            assert 'format' in attrs
