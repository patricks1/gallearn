"""
sha256-based dataset locking.

A training HDF5's row order isn't guaranteed stable across rebuilds
under the same filename (src/Dataset.jl assembles its file list from
unsorted directory listings), so a test lock, split file, or
checkpoint that references a dataset by filename alone has no way to
detect that the file's actual content has silently changed underneath
it. This module closes that gap: `lock_dataset` records a dataset
file's sha256 the first time it becomes load-bearing (referenced by a
test lock or split), and `verify_dataset` re-checks that hash on
every later use, so gallearn.splitting and gallearn.train can refuse
to run against a dataset that was never locked or that has drifted
since locking.

Lock files live under HASHES_DIR and are committed to the repo, like
gallearn.splitting.SPLITS_DIR's split files. lock_dataset never
overwrites an existing lock: once a dataset filename is locked, that
name is permanently tied to the content it had at lock time, and a
genuinely different dataset needs a new filename.
"""
import datetime
import hashlib
import json
import os
import pathlib

from . import config

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HASHES_DIR = REPO_ROOT / 'dataset_hashes'

_CHUNK_SIZE = 8 * 1024 * 1024


def _dataset_path(dataset_fname):
    return pathlib.Path(
        config.config['gallearn_paths']['project_data_dir']
    ) / dataset_fname


def _lock_path(dataset_fname):
    return HASHES_DIR / '{0}.json'.format(dataset_fname)


def compute_sha256(path):
    """
    Compute the sha256 of a file, reading it in fixed-size chunks so
    a multi-gigabyte HDF5 never has to fit in memory at once.

    Parameters
    ----------
    path : str or pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hex-encoded sha256 digest.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


def lock_dataset(dataset_fname):
    """
    Compute dataset_fname's sha256 and record it in HASHES_DIR.
    lock_dataset raises FileExistsError if a lock already exists for
    dataset_fname, since overwriting it would let the same filename
    silently point at different content. A genuinely different
    dataset needs its own filename instead of reusing a locked one.

    Parameters
    ----------
    dataset_fname : str
        Dataset filename (resolved against
        config.config['gallearn_paths']['project_data_dir'], the
        same way gallearn.preprocessing.load_metadata resolves it).

    Returns
    -------
    dict
        The lock record written to HASHES_DIR / '<dataset_fname>.json'
        ('dataset_fname', 'sha256', 'locked_at', 'size_bytes').
    """
    lock_path = _lock_path(dataset_fname)
    if lock_path.exists():
        raise FileExistsError(
            '{0} is already locked at {1}. A dataset filename is'
            ' permanently tied to the content it had when locked;'
            ' give a genuinely different dataset a new filename'
            ' instead of relocking this one.'.format(
                dataset_fname, lock_path
            )
        )

    dataset_path = _dataset_path(dataset_fname)
    print('Hashing {0}...'.format(dataset_path))
    sha256 = compute_sha256(dataset_path)
    lock = {
        'dataset_fname': dataset_fname,
        'sha256': sha256,
        'locked_at': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'size_bytes': os.path.getsize(dataset_path),
    }

    HASHES_DIR.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'w') as f:
        json.dump(lock, f, indent=2, sort_keys=True)
        f.write('\n')
    print('Locked {0} as {1}.'.format(dataset_fname, lock_path))
    return lock


def verify_dataset(dataset_fname):
    """
    Recompute dataset_fname's sha256 and check it against the lock
    HASHES_DIR holds for it. verify_dataset raises FileNotFoundError
    if dataset_fname has no lock yet (run scripts/lock_dataset.py
    first) and ValueError if the current file's hash disagrees with
    the locked one (the dataset's content has changed since locking).

    Parameters
    ----------
    dataset_fname : str
        Dataset filename (resolved against
        config.config['gallearn_paths']['project_data_dir']).

    Returns
    -------
    dict
        The matching lock record (see lock_dataset's return value).
    """
    lock_path = _lock_path(dataset_fname)
    if not lock_path.exists():
        raise FileNotFoundError(
            '{0} has no dataset lock at {1}. Run'
            ' `scripts/lock_dataset.py --dataset {0}` before using'
            ' it with scripts/split.py or scripts/train.py.'.format(
                dataset_fname, lock_path
            )
        )
    with open(lock_path) as f:
        lock = json.load(f)

    dataset_path = _dataset_path(dataset_fname)
    print('Verifying {0} against its lock...'.format(dataset_path))
    current_sha256 = compute_sha256(dataset_path)
    if current_sha256 != lock['sha256']:
        raise ValueError(
            '{0} no longer matches its lock at {1}. Locked sha256:'
            ' {2}. Current sha256: {3}. The dataset\'s content has'
            ' changed since it was locked, so any split or'
            ' checkpoint built against the locked version may no'
            ' longer refer to the same rows.'.format(
                dataset_fname,
                lock_path,
                lock['sha256'],
                current_sha256,
            )
        )
    return lock
