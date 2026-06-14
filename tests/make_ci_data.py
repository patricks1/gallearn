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


def make_sfr_data(seed=42):
    '''
    Sample 10 random galaxies from the full SFR dataset HDF5 and write them
    to tests/test_data/ under the same filename. Returns the galaxy IDs and
    orientations of the sampled rows so make_shapes_data can build matching
    shape CSVs.

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
    test_data_fname = dataset_fname#.replace('data', 'testdata')

    with h5py.File(PROJ_DATA_DIR / dataset_fname, 'r') as f_source:
        keys = f_source.keys()
        with h5py.File(TEST_DATA_DIR / test_data_fname, 'w') as f_test:
            for i, key in enumerate(keys):
                if i == 0:
                    # With the new nchw arranged h5 files, samples are in the
                    # rows (axis 0)
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

    Parameters
    ----------
    ids: np.ndarray of int
        Galaxy IDs from make_sfr_data.
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
                # 'left' preserves all rows from `shapes_df`.
                how='left',
                # Creates a new `_merge` column that specifies the
                # DataFrames in which
                # `merge` found the keys.
                indicator=True
            )
            # Only keep rows that were not in `key_df`.
            .query("_merge == 'left_only'")
            # Remove the indicator column.
            .drop(columns='_merge')
        )
        df_extras = df_antikey.sample(5, replace=False, random_state=seed)

        df_sample = shapes_df.merge(
            key_df,
            on=['galaxyID', 'view'],
            # 'inner' keeps only rows present in both DataFrames, i.e. the
            # shapes rows that match the requested galaxy-orientation pairs.
            how='inner',
            validate='one_to_one'
        )

        # Inject a few shapes that are not in the requested id-orientations
        df_sample = pd.concat((df_sample, df_extras), axis=0)

        df_sample.to_csv(TEST_DATA_DIR / fname, index=False)
        return shapes_df

    sample_df('host_2d_shapes', ids, orientations)
    sample_df('sat_2d_shapes', ids, orientations)
    sample_df('octant_shapes', ids, orientations)

    return None


def make_octant_image_files(gal_ids=(112, 910)):
    '''
    Build downsampled (64x64) octant HDF5 fixtures for the given galaxy
    IDs and write them to tests/test_data/octant_images/. Then run
    _process_galaxy on each fixture and save the Sersic fit results to
    tests/test_data/octant_shapes_reference.json as the stored regression
    reference, keyed by galaxy ID then projection name.

    FOV is kept at its original value -- each galaxy spans the same
    physical extent in kpc regardless of pixel resolution.

    Parameters
    ----------
    gal_ids : tuple of int, default (112, 910)
        Galaxy IDs to process.
    '''
    import json
    import queue as queue_mod
    import scipy.ndimage
    from gallearn import gen_octant_shapes

    octant_img_dir = pathlib.Path(
        config.config['gallearn_paths']['octant_img_dir']
    )
    out_dir = TEST_DATA_DIR / 'octant_images'
    out_dir.mkdir(parents=True, exist_ok=True)
    target_pixels = 64
    ref = {}

    for gal_id in gal_ids:
        # Galaxy IDs are the second underscore-delimited token in each
        # filename (e.g. object_112_host_ugrband_FOV17_p850.hdf5 -> 112).
        matches = sorted(
            octant_img_dir.glob(f'object_{gal_id}_*_ugrband_*.hdf5')
        )
        if not matches:
            raise FileNotFoundError(
                f'No HDF5 file for galaxy {gal_id} found in {octant_img_dir}'
            )
        src = matches[0]
        dst = out_dir / src.name

        with h5py.File(src, 'r') as f_src, h5py.File(dst, 'w') as f_dst:
            first_proj = gen_octant_shapes.OCTANT_PROJECTIONS[0]
            orig_pixels = int(f_src[first_proj].attrs['pixels'])
            zoom = target_pixels / orig_pixels
            for proj in gen_octant_shapes.OCTANT_PROJECTIONS:
                # Copy all attributes, then override 'pixels' to reflect
                # the downsampled resolution. FOV stays at its original
                # value.
                grp = f_dst.create_group(proj)
                for k, v in f_src[proj].attrs.items():
                    grp.attrs[k] = v
                grp.attrs['pixels'] = target_pixels
                # Downsample each band (u, g, r) to target_pixels x
                # target_pixels. scipy.ndimage.zoom rescales the array by
                # the given factor (< 1 shrinks, > 1 enlarges) using
                # spline interpolation (order 3 by default).
                for band in f_src[proj].keys():
                    arr = f_src[proj][band][()]
                    small = scipy.ndimage.zoom(arr, zoom).astype(np.float32)
                    grp.create_dataset(band, data=small)

        # Run the actual fitting code on the downsampled fixture and store
        # the results as the regression reference. Future test runs compare
        # against this JSON to catch unintended changes to _process_galaxy.
        rows = gen_octant_shapes._process_galaxy(
            (gal_id, str(dst), queue_mod.SimpleQueue())
        )
        ref[str(gal_id)] = {
            r['view']: {
                k: v for k, v in r.items()
                if k not in ('galaxyID', 'FOV', 'pixel', 'view', 'band')
            }
            for r in rows
        }

    ref_path = TEST_DATA_DIR / 'octant_shapes_reference.json'
    with open(ref_path, 'w') as fh:
        json.dump(ref, fh, indent=2)

    return None


def make_image_files():
    """Create minimal per-galaxy HDF5 image files with attributes
    matching Courtney's format.  No actual image data is stored;
    only the projection group attributes are needed for
    scan_image_dirs and tests."""
    import gallearn

    config_paths = gallearn.config.config['gallearn_paths']
    host_image_dir = pathlib.Path(
        config_paths['host_image_dir']
    )
    sat_image_dir = pathlib.Path(
        config_paths['sat_image_dir']
    )

    # Copy attributes from the real image files for the two test
    # galaxies.  Only attributes are needed; image data is omitted.
    test_files = [
        (
            host_image_dir
            / 'object_768_host_ugrband_FOV15_p750.hdf5',
            TEST_DATA_DIR / 'host_band_ugr',
        ),
        (
            sat_image_dir
            / 'object_1271_sate_ugrband_FOV13_p650.hdf5',
            TEST_DATA_DIR / 'sat_band_ugr',
        ),
    ]

    for src_path, dst_dir in test_files:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / src_path.name
        with h5py.File(src_path, 'r') as src:
            grp_name = list(src.keys())[0]
            attrs = dict(src[grp_name].attrs)
        with h5py.File(dst_path, 'w') as dst:
            for proj in ['xy', 'yz', 'zx']:
                grp = dst.create_group(
                    f'projection_{proj}'
                )
                for k, v in attrs.items():
                    grp.attrs[k] = v
                grp.attrs['projection'] = proj

    return None


def make_bound_particle_filters():
    """Create test bound particle filter files by intersecting
    the real bound particle IDs with the particle IDs present
    in the downsampled test particle files."""
    import gallearn

    firebox_data_dir = pathlib.Path(
        gallearn.config.config['gallearn_paths'][
            'firebox_data_dir'
        ]
    )
    test_objects_dir = (
        TEST_DATA_DIR / 'objects_1200_original'
    )

    for gal_id in [768, 1271]:
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
            print(
                f'Warning: {src_ahf_path} not found,'
                f' skipping.'
            )
            continue

        with h5py.File(src_ahf_path, 'r') as ahf:
            bound_ids = ahf['particleIDs'][:]
            bound_types = ahf['partTypes'][:]

        with h5py.File(test_particles_path, 'r') as p:
            test_gas_ids = set(p['gas_id'][:].astype(int))
            test_star_ids = set(
                p['stellar_id'][:].astype(int)
            )

        test_ids = test_gas_ids | test_star_ids
        keep = np.array([
            int(pid) in test_ids
            for pid in bound_ids
        ])

        with h5py.File(dst_path, 'w') as out:
            out.create_dataset(
                'particleIDs', data=bound_ids[keep],
            )
            out.create_dataset(
                'partTypes', data=bound_types[keep],
            )

    return None


def make_firebox_data():
    import uci_tools
    import gallearn

    firebox_data_dir = pathlib.Path(
        gallearn.config.config['paths']['firebox_data_dir']
    )
    output_dir = TEST_DATA_DIR / 'objects_1200_original'
    output_dir.mkdir(parents=False, exist_ok=True)

    for fname in [
            'particles_within_Rvir_object_768.hdf5',
            'particles_within_Rvir_object_1271.hdf5']:
        orig_path = (
            firebox_data_dir
            / 'objects_1200_original'
            / fname
        )
        output_path = output_dir / fname
        
        grps = uci_tools.tools.get_downsample_groups(orig_path, 3)
        uci_tools.tools.downsample_data(orig_path, output_path, grps, 1.e-2)

    return None


if __name__ == '__main__':
    ids, orientations = make_sfr_data()
    make_shapes_data(ids, orientations)
    make_image_files()
    make_octant_image_files()
    make_firebox_data()
    make_bound_particle_filters()
