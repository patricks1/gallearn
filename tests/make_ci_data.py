import h5py
import pathlib
import numpy as np
import pandas as pd

from gallearn import config

PROJ_DATA_DIR = pathlib.Path(
    config.config['gallearn_paths']['project_data_dir']
)

TEST_DATA_DIR = pathlib.Path(__file__).parent / 'test_data'
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def pick_fixture_ids(ids, n=2):
    '''
    Return the first n galaxy IDs from ids that have all required file types:
    host/sat image HDF5s, octant image HDF5s, and firebox particle files.
    Raises RuntimeError if fewer than n qualifying IDs are found.

    Parameters
    ----------
    ids: array-like of int
        Candidate galaxy IDs (e.g. from make_sfr_data).
    n: int, default 2
        Number of qualifying IDs to return.

    Returns
    -------
    list of int
        The first n IDs with all required files.
    '''
    host_image_dir = pathlib.Path(
        config.config['gallearn_paths']['host_image_dir']
    )
    sat_image_dir = pathlib.Path(
        config.config['gallearn_paths']['sat_image_dir']
    )
    octant_img_dir = pathlib.Path(
        config.config['gallearn_paths']['octant_img_dir']
    )
    firebox_data_dir = pathlib.Path(
        config.config['gallearn_paths']['firebox_data_dir']
    )
    objects_dir = firebox_data_dir / 'objects_1200_original'

    chosen = []
    for gal_id in ids:
        has_host = bool(list(host_image_dir.glob(
            f'object_{gal_id}_host_ugrband_*.hdf5'
        )))
        has_sat = bool(list(sat_image_dir.glob(
            f'object_{gal_id}_sate_ugrband_*.hdf5'
        )))
        has_octant = bool(list(octant_img_dir.glob(
            f'object_{gal_id}_*_ugrband_*.hdf5'
        )))
        has_firebox = (
            objects_dir
            / f'particles_within_Rvir_object_{gal_id}.hdf5'
        ).exists()
        if has_host and has_sat and has_octant and has_firebox:
            chosen.append(gal_id)
        if len(chosen) == n:
            break

    if len(chosen) < n:
        raise RuntimeError(
            f'Only {len(chosen)} galaxy IDs with all required file types'
            f' found in the {len(ids)} candidates; need {n}.'
        )
    return chosen


def make_sfr_data(seed=42):
    '''
    Sample 10 random galaxies from the full SFR dataset HDF5 and write them
    to tests/test_data/ under the same filename. Returns the galaxy IDs and
    orientations of the sampled rows so make_shapes_data can build matching
    shape CSVs.

    Consumed by: no direct test; provides IDs that flow into all other
    make_* functions via pick_fixture_ids.

    Parameters
    ----------
    seed: int, default 42
        Random seed for reproducible sampling.

    Returns
    -------
    ids: np.ndarray of int
        Galaxy IDs of the 10 sampled rows.
    orientations: np.ndarray of str
        Projection names of the 10 sampled rows.
    '''
    from gallearn import preprocessing

    dataset_fname = (
        'gallearn_data_256x256_11proj_wsat_wvmap_avg_sfr_tgt_mp.h5'
    )
    test_data_fname = dataset_fname

    with h5py.File(PROJ_DATA_DIR / dataset_fname, 'r') as f_source:
        keys = f_source.keys()
        with h5py.File(TEST_DATA_DIR / test_data_fname, 'w') as f_test:
            for i, key in enumerate(keys):
                if i == 0:
                    N = f_source[key].shape[0]
                    rng = np.random.default_rng(seed)
                    idxs = rng.choice(N, size=10, replace=False)

                source_literal = f_source[key][:]
                test_literal = source_literal[idxs]
                f_test.create_dataset(key, data=test_literal)

                if key == 'obs_sorted':
                    ids = np.array([
                        int(obj.decode('utf-8').replace('object_', ''))
                        for obj in test_literal
                    ])
                elif key == 'orientations':
                    orientations = np.array([
                        orientation.decode('utf-8')
                        for orientation in test_literal
                    ])
    return ids, orientations


def make_shapes_data(ids, orientations, seed=42):
    '''
    Build trimmed versions of the host, satellite, and octant Sersic shape
    CSVs for CI. Each full CSV is sampled down to the rows matching the given
    galaxy IDs and orientations, plus 5 randomly sampled non-matching rows as
    decoys. Writes one CSV per source file to tests/test_data/ under the same
    filename as the source.

    Consumed by: tests/test_preprocessing.py (host/sat shapes CSVs),
    tests/test_gen_octant_shapes.py (octant shapes CSV via config).

    Parameters
    ----------
    ids: array-like of int
        Galaxy IDs from make_sfr_data or pick_fixture_ids.
    orientations: np.ndarray of str
        Projection names from make_sfr_data.
    seed: int, default 42
        Random seed for reproducible sampling.

    Returns
    -------
    None
    '''
    def sample_df(config_key, ids, orientations):
        '''
        Read the shapes CSV at config_key, keep the rows matching ids and
        orientations, append 5 randomly sampled non-matching rows as decoys,
        and write the result to tests/test_data/.
        '''
        key_df = pd.DataFrame({'galaxyID': ids, 'view': orientations})
        assert not key_df.duplicated().any(), (
            'You have provided duplicate galaxy-orientation samples.'
        )

        path = pathlib.Path(config.config['gallearn_paths'][config_key])
        fname = path.name
        dtype_dict = {'galaxyID': int}
        shapes_df = pd.read_csv(path, dtype=dtype_dict)
        # We can only have one radius per galaxy, so we need to choose the
        # r band. Otherwise there are duplicates of galaxyID-view.
        shapes_df = shapes_df.loc[shapes_df['band'] == 'band_r']

        df_antikey = (
            shapes_df
            .merge(
                key_df,
                on=['galaxyID', 'view'],
                how='left',
                indicator=True,
            )
            .query("_merge == 'left_only'")
            .drop(columns='_merge')
        )
        df_extras = df_antikey.sample(5, replace=False, random_state=seed)

        df_sample = shapes_df.merge(
            key_df,
            on=['galaxyID', 'view'],
            how='inner',
            validate='one_to_one',
        )

        df_sample = pd.concat((df_sample, df_extras), axis=0)
        df_sample.to_csv(TEST_DATA_DIR / fname, index=False)
        return shapes_df

    sample_df('host_2d_shapes', ids, orientations)
    sample_df('sat_2d_shapes', ids, orientations)
    sample_df('octant_shapes', ids, orientations)

    return None


def make_scan_dir_stubs(ids):
    '''
    Create minimal per-galaxy HDF5 image files with attributes matching the
    production format. No actual image data is stored; only projection group
    attributes are needed for scan_image_dirs. One host file and one sat file
    are written per galaxy ID.

    Consumed by: tests/test_gen_octant_images.py (TestScanImageDirs).

    Parameters
    ----------
    ids: list of int
        Galaxy IDs to create stubs for.

    Returns
    -------
    None
    '''
    host_image_dir = pathlib.Path(
        config.config['gallearn_paths']['host_image_dir']
    )
    sat_image_dir = pathlib.Path(
        config.config['gallearn_paths']['sat_image_dir']
    )

    for gal_id in ids:
        host_matches = sorted(host_image_dir.glob(
            f'object_{gal_id}_host_ugrband_*.hdf5'
        ))
        sat_matches = sorted(sat_image_dir.glob(
            f'object_{gal_id}_sate_ugrband_*.hdf5'
        ))
        file_pairs = []
        if host_matches:
            file_pairs.append(
                (host_matches[0], TEST_DATA_DIR / 'host_band_ugr')
            )
        if sat_matches:
            file_pairs.append(
                (sat_matches[0], TEST_DATA_DIR / 'sat_band_ugr')
            )

        for src_path, dst_dir in file_pairs:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / src_path.name
            with h5py.File(src_path, 'r') as src:
                grp_name = list(src.keys())[0]
                attrs = dict(src[grp_name].attrs)
            with h5py.File(dst_path, 'w') as dst:
                for proj in ['xy', 'yz', 'zx']:
                    grp = dst.create_group(f'projection_{proj}')
                    for k, v in attrs.items():
                        grp.attrs[k] = v
                    grp.attrs['projection'] = proj

    return None


def make_octant_image_files(ids):
    '''
    Build downsampled (64x64) octant HDF5 fixtures for the given galaxy IDs
    and write them to tests/test_data/octant_images/. Then run _process_galaxy
    on each fixture and save the Sersic fit results to
    tests/test_data/octant_shapes_reference.csv as the stored regression
    reference (flat CSV with columns: galaxyID, FOV, pixel, view, band, b_a,
    PA, n, Re, Ie).

    FOV is kept at its original value -- each galaxy spans the same physical
    extent in kpc regardless of pixel resolution. make_octant_image_files
    overwrites the 'pixels' HDF5 attribute to 64. Therefore, note that the
    pixel count in these fixtures differs from the original production files.

    Consumed by: tests/test_gen_octant_shapes.py
    (test_process_galaxy_regression, test_resume_partial_galaxy).

    Parameters
    ----------
    ids: list of int
        Galaxy IDs to process.

    Returns
    -------
    None
    '''
    import queue as queue_mod
    import scipy.ndimage
    from gallearn import gen_octant_shapes

    octant_img_dir = pathlib.Path(
        config.config['gallearn_paths']['octant_img_dir']
    )
    out_dir = TEST_DATA_DIR / 'octant_images'
    out_dir.mkdir(parents=True, exist_ok=True)
    target_pixels = 64
    all_rows = []

    for gal_id in ids:
        matches = sorted(
            octant_img_dir.glob(f'object_{gal_id}_*_ugrband_*.hdf5')
        )
        if not matches:
            raise FileNotFoundError(
                f'No HDF5 file for galaxy {gal_id} in {octant_img_dir}'
            )
        src = matches[0]
        dst = out_dir / src.name

        with h5py.File(src, 'r') as f_src, h5py.File(dst, 'w') as f_dst:
            first_proj = gen_octant_shapes.OCTANT_PROJECTIONS[0]
            orig_pixels = int(f_src[first_proj].attrs['pixels'])
            zoom = target_pixels / orig_pixels
            for proj in gen_octant_shapes.OCTANT_PROJECTIONS:
                grp = f_dst.create_group(proj)
                for k, v in f_src[proj].attrs.items():
                    grp.attrs[k] = v
                grp.attrs['pixels'] = target_pixels
                for band in f_src[proj].keys():
                    arr = f_src[proj][band][()]
                    small = scipy.ndimage.zoom(
                        arr, zoom
                    ).astype(np.float32)
                    grp.create_dataset(band, data=small)

        q = queue_mod.SimpleQueue()
        gen_octant_shapes._process_galaxy(
            (gal_id, str(dst), q, frozenset())
        )
        while not q.empty():
            kind, _, payload = q.get()
            if kind == 'row':
                all_rows.append(payload)

    ref_path = TEST_DATA_DIR / 'octant_shapes_reference.csv'
    pd.DataFrame(all_rows).to_csv(ref_path, index=False)

    return None


def make_bound_particle_filters(ids):
    '''
    Create test bound particle filter files by intersecting the real bound
    particle IDs with the particle IDs present in the downsampled test
    particle files.

    Consumed by: tests/test_preprocessing.py (bound particle filter tests).

    Parameters
    ----------
    ids: list of int
        Galaxy IDs to create filter files for.

    Returns
    -------
    None
    '''
    firebox_data_dir = pathlib.Path(
        config.config['gallearn_paths']['firebox_data_dir']
    )
    test_objects_dir = TEST_DATA_DIR / 'objects_1200_original'

    for gal_id in ids:
        src_ahf_path = (
            firebox_data_dir
            / 'objects_1200_original'
            / f'bound_particle_filters_object_{gal_id}.hdf5'
        )
        test_particles_path = (
            test_objects_dir
            / f'particles_within_Rvir_object_{gal_id}.hdf5'
        )
        dst_path = (
            test_objects_dir
            / f'bound_particle_filters_object_{gal_id}.hdf5'
        )

        if not src_ahf_path.exists():
            print(f'Warning: {src_ahf_path} not found, skipping.')
            continue

        with h5py.File(src_ahf_path, 'r') as ahf:
            bound_ids = ahf['particleIDs'][:]
            bound_types = ahf['partTypes'][:]

        with h5py.File(test_particles_path, 'r') as p:
            test_gas_ids = set(p['gas_id'][:].astype(int))
            test_star_ids = set(p['stellar_id'][:].astype(int))

        test_ids = test_gas_ids | test_star_ids
        keep = np.array([int(pid) in test_ids for pid in bound_ids])

        with h5py.File(dst_path, 'w') as out:
            out.create_dataset('particleIDs', data=bound_ids[keep])
            out.create_dataset('partTypes', data=bound_types[keep])

    return None


def make_firebox_data(ids):
    '''
    Downsample firebox particle HDF5 files to 1% of particles for CI speed
    and write them to tests/test_data/objects_1200_original/.

    Consumed by: tests/test_preprocessing.py (firebox particle tests).

    Parameters
    ----------
    ids: list of int
        Galaxy IDs to downsample.

    Returns
    -------
    None
    '''
    import uci_tools
    import gallearn

    firebox_data_dir = pathlib.Path(
        gallearn.config.config['gallearn_paths']['firebox_data_dir']
    )
    output_dir = TEST_DATA_DIR / 'objects_1200_original'
    output_dir.mkdir(parents=False, exist_ok=True)

    for gal_id in ids:
        fname = f'particles_within_Rvir_object_{gal_id}.hdf5'
        orig_path = firebox_data_dir / 'objects_1200_original' / fname
        output_path = output_dir / fname
        grps = uci_tools.tools.get_downsample_groups(orig_path, 3)
        uci_tools.tools.downsample_data(orig_path, output_path, grps, 1.e-2)

    return None


if __name__ == '__main__':
    ids, orientations = make_sfr_data()
    fixture_ids = pick_fixture_ids(ids, n=2)
    make_shapes_data(fixture_ids, orientations)
    make_scan_dir_stubs(fixture_ids)
    make_octant_image_files(fixture_ids)
    make_firebox_data(fixture_ids)
    make_bound_particle_filters(fixture_ids)
