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


def make_sfr_data():
    from gallearn import preprocessing

    dataset_fname = 'gallearn_data_256x256_3proj_wsat_wvmap_avg_sfr_tgt.h5'
    test_data_fname = dataset_fname#.replace('data', 'testdata')

    with h5py.File(PROJ_DATA_DIR / dataset_fname, 'r') as f_source:
        keys = f_source.keys()
        with h5py.File(TEST_DATA_DIR / test_data_fname, 'w') as f_test:
            for i, key in enumerate(keys):
                if i == 0:
                    N = f_source[key].shape[-1]
                    rng = np.random.default_rng(42)
                    idxs = rng.choice(N, size=10, replace=False)

                source_literal = f_source[key][:]
                test_literal = source_literal[..., idxs]
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

        path = pathlib.Path(config.config['gallearn_paths'][config_key])
        fname = path.name
        df = pd.read_csv(path)

        df_sample = df.merge(key_df, on=['galaxyID', 'view'], how='inner')
        df_sample.to_csv(TEST_DATA_DIR / fname)
        return df

    sample_df('host_2d_shapes', ids, orientations)
    sample_df('sat_2d_shapes', ids, orientations)

    return None


if __name__ == '__main__':
    ids, orientations = make_sfr_data()
    make_shapes_data(ids, orientations)
