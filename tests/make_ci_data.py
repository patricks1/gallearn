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
    from gallearn import preprocessing

    dataset_fname = (
        'gallearn_data_256x256_3proj_wsat_wvmap_avg_sfr_tgt_nchw.h5'
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
    def sample_df(config_key, ids, orientations):
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
            how='inner',
            validate='one_to_one'
        )

        # Inject a few shapes that are not in the requested id-orientations
        df_sample = pd.concat((df_sample, df_extras), axis=0)

        df_sample.to_csv(TEST_DATA_DIR / fname, index=False)
        return shapes_df

    sample_df('host_2d_shapes', ids, orientations)
    sample_df('sat_2d_shapes', ids, orientations)

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
    make_firebox_data()
    make_bound_particle_filters()
