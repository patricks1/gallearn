import numpy as np

import h5py

from gallearn import config
from gallearn import preprocessing


def _write_dataset(path, n_rows=3, attrs=None):
    '''Write a minimal dataset HDF5 carrying the fields load_metadata
    reads, optionally with root attributes naming its target.'''
    with h5py.File(path, 'w') as f:
        f.create_dataset(
            'X', data=np.zeros((n_rows, 4, 2, 2), dtype=np.float32)
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
            data=np.linspace(0., 0.5, n_rows).reshape(-1, 1),
        )
        f.create_dataset(
            'Re', data=np.ones((n_rows, 1), dtype=np.float64)
        )
        for key, value in (attrs or {}).items():
            f.attrs[key] = value


def test_load_metadata_reads_the_target_attribute(
        tmp_path,
        monkeypatch):
    '''Verify that a dataset declaring its own target reports it, so
    training need not be told what the file already records.'''
    monkeypatch.setitem(
        config.config['gallearn_paths'],
        'project_data_dir',
        str(tmp_path),
    )
    _write_dataset(
        tmp_path / 'ds.h5',
        attrs={
            'tgt_type': 'fgas',
            'tgt_source': 'firebox_summary_stats.csv',
        },
    )
    d, N, _ = preprocessing.load_metadata('ds.h5')
    assert d['tgt_type'] == 'fgas'
    assert N == 3


def test_load_metadata_reports_no_target_for_older_datasets(
        tmp_path,
        monkeypatch):
    '''Verify that a dataset built before the target attribute
    existed reports None rather than guessing, which is what forces
    the caller to name the target explicitly.'''
    monkeypatch.setitem(
        config.config['gallearn_paths'],
        'project_data_dir',
        str(tmp_path),
    )
    _write_dataset(tmp_path / 'ds.h5')
    d, _, _ = preprocessing.load_metadata('ds.h5')
    assert d['tgt_type'] is None
