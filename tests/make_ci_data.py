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


def make_shapes_data(ids, orientations):
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
        df_extras = df_antikey.sample(5, replace=False)

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


if __name__ == '__main__':
    ids, orientations = make_sfr_data()
    make_shapes_data(ids, orientations)
