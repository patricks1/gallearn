'''
End-to-end smoke test for gallearn.train.main().

The unit tests in test_target_dispatch.py exercise prepare_targets
and compute_valid_indices directly, with every argument spelled out
by hand. That style previously missed a real bug: main()'s fresh-run
branch called prepare_targets() without tgt_type, a TypeError that
only shows up by actually calling main() the way scripts/train.py
does. This file runs main() itself, on a tiny synthetic dataset, so
a caller-side wiring mistake like that fails a test instead of only
showing up on a real training run.
'''
import json

import numpy as np
import pytest

import h5py

from gallearn import config
from gallearn import dataset_lock
from gallearn import train


def _write_dataset(path, tgt_type, n_rows=6, res=32):
    '''Write a minimal but real-shaped dataset HDF5: 4-channel images
    (3 bands + vmap), Re, and a target attrs-declared so main() needs
    no --target. Values are non-constant so per-channel scaling
    stats come out finite.'''
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        f.create_dataset(
            'X',
            data=rng.normal(
                size=(n_rows, 4, res, res)
            ).astype(np.float32),
        )
        f.create_dataset(
            'obs_sorted',
            data=np.array(
                ['object_{0}'.format(i) for i in range(n_rows)],
                dtype='S20',
            ),
        )
        f.create_dataset(
            'orientations',
            data=np.array(['p000'] * n_rows, dtype='S20'),
        )
        f.create_dataset(
            'file_names',
            data=np.array(['f.hdf5'] * n_rows, dtype='S20'),
        )
        f.create_dataset(
            'ys_sorted',
            data=rng.uniform(0., 0.9, size=(n_rows, 1)),
        )
        f.create_dataset(
            'Re', data=np.ones((n_rows, 1), dtype=np.float64)
        )
        f.attrs['tgt_type'] = tgt_type
        f.attrs['tgt_source'] = 'firebox_summary_stats.csv'


@pytest.mark.parametrize('tgt_type', ['fgas', 'fdm'])
def test_main_runs_a_fresh_regressor_epoch(
        tgt_type, tmp_path, monkeypatch):
    '''Verify a fresh (non-resume) regressor run completes one
    epoch end to end: dataset load, target/scaling prep, training and
    validation loops, and a checkpoint write. This is the exact path
    that broke when prepare_targets() was called without tgt_type.
    Covers both fraction targets, since they share the same
    read_halo_stats_tgt() source file on the Julia side and this is
    what verifies that rename/merge didn't regress either one.'''
    monkeypatch.setitem(
        config.config['gallearn_paths'],
        'project_data_dir',
        str(tmp_path),
    )
    monkeypatch.setattr(dataset_lock, 'HASHES_DIR', tmp_path / 'hashes')

    dataset_fname = 'ds.h5'
    _write_dataset(tmp_path / dataset_fname, tgt_type)
    dataset_lock.lock_dataset(dataset_fname)

    n_rows = 6
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps({
        'train_galaxies': [
            'object_{0}'.format(i) for i in range(n_rows - 2)
        ],
        'val_galaxies': [
            'object_{0}'.format(i)
            for i in range(n_rows - 2, n_rows)
        ],
    }))

    run_name = 'ci_smoke_{0}'.format(tgt_type)
    model = train.main(
        task='regressor',
        model_type='standard',
        split_file_path=str(split_path),
        dataset=dataset_fname,
        run_name=run_name,
        n_epochs=1,
        batch_size=2,
        wandb_mode='n',
    )

    assert model is not None
    checkpoints = list(
        (tmp_path / run_name).glob('checkpoint_epoch*.pt')
    )
    assert len(checkpoints) == 1
