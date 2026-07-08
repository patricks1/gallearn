import h5py
import pathlib
import numpy as np
import pandas as pd

from gallearn import config

PROJ_DATA_DIR = pathlib.Path(
    config.config['gallearn_paths']['project_data_dir']
)
FIREBOX_SNAP = config.config['gallearn_paths']['firebox_snap']

TEST_DATA_DIR = pathlib.Path(__file__).parent / 'test_data'
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def pick_fixture_ids(ids, n=2):
    '''
    Return the first n unique galaxy IDs from ids that have all required file
    types: a host or sat image HDF5, octant image HDF5s, and firebox particle
    files. Deduplicates ids before iterating so a galaxy that appears more
    than once in ids (e.g. sampled with different projections) is only
    considered once. Raises RuntimeError if fewer than n qualifying IDs are
    found.

    Parameters
    ----------
    ids: array-like of int
        Candidate galaxy IDs (e.g. from make_sfr_data); may contain
        duplicates.
    n: int, default 2
        Number of qualifying IDs to return.

    Returns
    -------
    list of int
        The first n unique IDs with all required files.
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
    objects_dir = firebox_data_dir / FIREBOX_SNAP

    chosen = []
    for gal_id in dict.fromkeys(ids):
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
        if (has_host or has_sat) and has_octant and has_firebox:
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
    Sample 10 random rows from the full SFR dataset HDF5 and write them to
    tests/test_data/ under the same filename. Returns the galaxy IDs of the
    sampled rows so the __main__ block can call pick_fixture_ids and
    make_shapes_data.

    The HDF5 this function writes is consumed by
    test_BernoulliNet.py::test_BernoulliNet via preprocessing.load_data().
    The __main__ block also passes the returned ids to pick_fixture_ids
    and make_shapes_data.

    The source dataset predates Dataset.jl writing an 'Re' key, so this
    function fabricates a synthetic 'Re' dataset (same seeded rng as the
    row sampling above) instead of copying one from the source. Once a
    production dataset built with the current Dataset.jl exists, this
    fabrication can be replaced by just letting the source-copy loop below
    pick up the real 'Re' key like every other key.

    Parameters
    ----------
    seed: int, default 42
        Random seed for reproducible sampling.

    Returns
    -------
    ids: np.ndarray of int
        Galaxy IDs of the 10 sampled rows. May contain duplicates when the
        same galaxy appears with more than one projection in the HDF5.
    orientations: np.ndarray of str
        Projection names of the 10 sampled rows. Returned alongside ids to
        make it visible that ids may be non-unique.
    '''
    dataset_fname = config.config['gallearn_paths']['dataset']
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
            if 'Re' not in f_source.keys():
                Re = rng.uniform(0.5, 5.0, size=(10, 1))
                f_test.create_dataset('Re', data=Re)
    return ids, orientations


def make_shapes_data(ids, seed=42):
    '''
    Build trimmed versions of the host, satellite, and octant Sersic shape
    CSVs for CI. Each full CSV is filtered to all rows whose galaxyID appears
    in ids (across all projections), plus 5 randomly sampled non-matching rows
    as decoys. Writes one CSV per source file to tests/test_data/ under the
    same filename as the source.

    Consumed by: test_gen_octant_shapes.py::test_resume_partial_galaxy, via
    gen_octant_shapes.gen(), which reads host_2d_shapes and sat_2d_shapes to
    build the eligible galaxy ID set. (Re used to come from a Python-side
    join against these three CSVs via cnn.get_radii(); that join now
    happens in Julia's Dataset.read_2d_shapes(), so test_BernoulliNet.py
    no longer depends on these CSVs, only on the 'Re' key make_sfr_data
    fabricates directly into the sampled HDF5.)

    Parameters
    ----------
    ids: array-like of int
        Galaxy IDs from make_sfr_data; may contain duplicates.
    seed: int, default 42
        Random seed for reproducible sampling.

    Returns
    -------
    None
    '''
    def sample_df(config_key, key_ids):
        '''
        Read the shapes CSV at config_key, keep all rows whose galaxyID is in
        key_ids, append 5 randomly sampled non-matching rows as decoys, and
        write the result to tests/test_data/.
        '''
        path = pathlib.Path(config.config['gallearn_paths'][config_key])
        fname = path.name
        dtype_dict = {'galaxyID': int}
        shapes_df = pd.read_csv(path, dtype=dtype_dict)
        # Filter to band_r so each (galaxyID, view) appears only once, which
        # is what Dataset.read_2d_shapes() requires for its join.
        shapes_df = shapes_df.loc[shapes_df['band'] == 'band_r']

        df_sample = shapes_df[shapes_df['galaxyID'].isin(key_ids)]
        df_extras = (
            shapes_df[~shapes_df['galaxyID'].isin(key_ids)]
            .sample(5, replace=False, random_state=seed)
        )

        df_sample = pd.concat((df_sample, df_extras), axis=0)
        df_sample.to_csv(TEST_DATA_DIR / fname, index=False)
        return shapes_df

    key_ids = set(ids)
    sample_df('host_2d_shapes', key_ids)
    sample_df('sat_2d_shapes', key_ids)
    sample_df('octant_shapes', key_ids)

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
    import shutil

    host_image_dir = pathlib.Path(
        config.config['gallearn_paths']['host_image_dir']
    )
    sat_image_dir = pathlib.Path(
        config.config['gallearn_paths']['sat_image_dir']
    )

    # Clear before repopulating so files for galaxies that were selected in a
    # previous run but are no longer selected don't linger as stale fixtures.
    for stub_dir in [
        TEST_DATA_DIR / 'host_band_ugr',
        TEST_DATA_DIR / 'sat_band_ugr',
    ]:
        if stub_dir.exists():
            shutil.rmtree(stub_dir)

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
    and write them to tests/test_data/octant_images/.

    FOV is kept at its original value -- each galaxy spans the same physical
    extent in kpc regardless of pixel resolution. make_octant_image_files
    overwrites the 'pixels' HDF5 attribute to 64, so the pixel count in
    these fixtures differs from the original production files.

    Consumed by: make_octant_shapes_reference and
    tests/test_gen_octant_shapes.py (test_resume_partial_galaxy).

    Parameters
    ----------
    ids: list of int
        Galaxy IDs to process.

    Returns
    -------
    None
    '''
    import scipy.ndimage
    from gallearn import gen_octant_shapes

    octant_img_dir = pathlib.Path(
        config.config['gallearn_paths']['octant_img_dir']
    )
    out_dir = TEST_DATA_DIR / 'octant_images'
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    target_pixels = 64

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

    return None


def make_octant_shapes_reference(ids, fitter):
    '''
    Run _process_galaxy on each fixture HDF5 in tests/test_data/octant_images/
    using the given fitter and write the results to a reference CSV.

    The reference CSV name encodes the fitter:
    - 'fit_sersic' -> octant_shapes_fitsersic_reference.csv
    - 'astrophot'  -> octant_shapes_astrophot_reference.csv

    Consumed by: tests/test_gen_octant_shapes.py
    (test_fit_sersic, test_astrophot).

    Parameters
    ----------
    ids: list of int
        Galaxy IDs whose fixture HDF5s to run _process_galaxy on.
    fitter: str
        Fitting backend passed to _process_galaxy. 'astrophot' calls
        AstroPhot's LM optimizer; 'fit_sersic' calls sersic_tools.fit_sersic
        from mockobservation_tools.

    Returns
    -------
    None
    '''
    import queue as queue_mod
    from gallearn import gen_octant_shapes

    out_dir = TEST_DATA_DIR / 'octant_images'
    if fitter == 'astrophot':
        ref_name = 'octant_shapes_astrophot_reference.csv'
    else:
        ref_name = 'octant_shapes_fitsersic_reference.csv'

    all_rows = []
    for gal_id in ids:
        matches = sorted(out_dir.glob(
            f'object_{gal_id}_*_ugrband_*.hdf5'
        ))
        if not matches:
            raise FileNotFoundError(
                f'No fixture HDF5 for galaxy {gal_id} in {out_dir}'
            )
        dst = matches[0]
        q = queue_mod.SimpleQueue()
        gen_octant_shapes._process_galaxy(
            (gal_id, str(dst), q, frozenset(), fitter)
        )
        while not q.empty():
            kind, _, payload = q.get()
            if kind == 'row':
                all_rows.append(payload)

    pd.DataFrame(all_rows).to_csv(TEST_DATA_DIR / ref_name, index=False)
    return None


def make_bound_particle_filters(ids):
    '''
    Build test bound particle filter files by intersecting the real bound
    particle IDs with the particle IDs present in the downsampled test
    particle files written by make_firebox_data.

    No test currently reads these files, so the __main__ block does not call
    this function. It exists for when a future test exercises the bound-only
    filtering path in gen_octant_images.process_galaxy (line 142), which
    passes these files to the image generator when present.

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
    # No clearing needed here: make_firebox_data (which must run first)
    # already wipes and recreates this directory.
    test_objects_dir = TEST_DATA_DIR / FIREBOX_SNAP

    for gal_id in ids:
        src_ahf_path = (
            firebox_data_dir
            / FIREBOX_SNAP
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
    and write them to tests/test_data/{FIREBOX_SNAP}/.

    No test currently reads these files, so the __main__ block does not call
    this function. The only script that would consume its output is
    make_bound_particle_filters, which also has no active test consumer.
    Call both functions from __main__ when a test exercises the firebox
    particle data path.

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
    output_dir = TEST_DATA_DIR / FIREBOX_SNAP
    # Clear before repopulating so files for galaxies that were selected in a
    # previous run but are no longer selected don't linger as stale fixtures.
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=False)

    for gal_id in ids:
        fname = f'particles_within_Rvir_object_{gal_id}.hdf5'
        orig_path = firebox_data_dir / FIREBOX_SNAP / fname
        output_path = output_dir / fname
        grps = uci_tools.tools.get_downsample_groups(orig_path, 3)
        uci_tools.tools.downsample_data(orig_path, output_path, grps, 1.e-2)

    return None


if __name__ == '__main__':
    ids, orientations = make_sfr_data()
    fixture_ids = pick_fixture_ids(ids, n=2)
    make_shapes_data(ids)
    make_scan_dir_stubs(fixture_ids)
    make_octant_image_files(fixture_ids)
    make_octant_shapes_reference(fixture_ids, fitter='fit_sersic')
    make_octant_shapes_reference(fixture_ids, fitter='astrophot')
