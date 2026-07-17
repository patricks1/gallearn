"""
Galaxy-level train/val splitting and a locked, representative test
set.

A "row" in the training HDF5 is one (galaxy, projection) pair. This
module always groups by galaxy: every projection of a galaxy lands
in the same split (the locked test set, train, or val), since a
galaxy's projections share mass, SFR, and correlated morphology.

There are two kinds of split files. Both live under SPLITS_DIR as
part of the repo:

- The test lock (see `update_test_lock`): the source of truth for
  which galaxies training must permanently exclude. It is monotonic:
  each version's locked galaxies are a strict superset of the
  previous version's, never a reassignment or removal.
  update_test_lock never edits an existing test lock file in place;
  each initial creation or later top-up instead writes a new,
  immutable `test_lock_v<N>.json` (see `latest_test_lock_path`), so
  a `test_lock_path` a split file records stays valid forever, and
  old lock states remain on disk as their own files rather than only
  recoverable through git history. Both initial creation and top-ups
  draw new galaxies with stratified sampling on stellar mass and
  sSFR, so the locked set tracks the population's mass/SFR
  distribution rather than a plain random draw.
- A train/val split (see `write_split`): an unstratified random
  split of whatever galaxies are not in the test lock. Multiple of
  these can coexist for different experiments.

`gallearn.train` never reads the test lock directly, only a
train/val split file, so the true test set never enters the training
loop.

Both `update_test_lock` and `write_split` require the dataset they
operate on to already be locked via `gallearn.dataset_lock`, since a
galaxy-to-row mapping computed against one version of a dataset file
is meaningless against a silently rebuilt one under the same
filename.
"""
import collections
import datetime
import json
import pathlib
import random

import numpy as np
import pandas as pd
import torch

from . import config
from . import dataset_lock
from . import preprocessing

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / 'splits'

# The filename mirrors read_sfr_tgt's fname in src/Dataset.jl, which
# also resolves its directory from project_data_dir (gallearn_dir).
# Not a separate config key, since scripts/gen_sfrs.jl writes only
# one such file per project_data_dir.
AVG_SFR_CSV = (
    pathlib.Path(config.config['gallearn_paths']['project_data_dir'])
    / 'avg_sfrs_1.0Gyr_no_bound_filter.csv'
)


def _now_iso():
    return datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()


def _load_json(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(data, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')


def load_avg_sfr_csv(csv_path=AVG_SFR_CSV):
    """
    Load a galaxy-mass lookup from the avg_sfrs CSV (see UCITools'
    ProcessFIREBox.get_avg_sfrs, which writes this file with an
    `id`, `grp_id`, `sfr`, `ssfr`, `Mstar` schema). `id` is a bare
    integer in the CSV; load_avg_sfr_csv prepends "object_" to
    match `obs_sorted`, the same transform src/Dataset.jl's
    `read_sfr_tgt` applies.

    Parameters
    ----------
    csv_path : str or pathlib.Path, optional
        Path to the avg_sfrs CSV. Defaults to AVG_SFR_CSV.

    Returns
    -------
    dict
        {galaxy_id: Mstar}, one entry per CSV row whose `id`
        parsed cleanly.
    """
    df = pd.read_csv(csv_path)
    masses = {}
    n_bad = 0
    for raw_id, mstar in zip(df['id'], df['Mstar']):
        try:
            galaxy_id = 'object_{0}'.format(int(raw_id))
        except (TypeError, ValueError):
            n_bad += 1
            continue
        masses[galaxy_id] = float(mstar)
    if n_bad > 0:
        print(
            'Skipped {0} rows in {1} with an unparseable galaxy'
            ' id.'.format(n_bad, csv_path)
        )
    return masses


def build_galaxy_index(obs_sorted):
    """
    Group HDF5 row indices by galaxy id.

    Parameters
    ----------
    obs_sorted : array-like of str
        The training HDF5's `obs_sorted` array (one galaxy id per
        row, e.g. "object_561"), sliced to the N rows in use.

    Returns
    -------
    dict
        {galaxy_id: [row_idx, ...]}, row_idx values in row order.
    """
    galaxy_index = collections.defaultdict(list)
    for row_idx, galaxy_id in enumerate(obs_sorted):
        galaxy_index[galaxy_id].append(row_idx)
    return dict(galaxy_index)


def galaxy_ssfr(galaxy_index, ys_sorted):
    """
    Collapse the per-row sSFR target down to one value per galaxy.
    sSFR is a property of the galaxy, not of the projection, so
    every one of a galaxy's projection rows in `ys_sorted` stores an
    independent copy of the same value. galaxy_ssfr asserts those
    per-row copies actually agree with each other rather than
    silently averaging over a disagreement, since disagreement would
    mean a data bug upstream.

    Parameters
    ----------
    galaxy_index : dict
        {galaxy_id: [row_idx, ...]}, from build_galaxy_index.
    ys_sorted : array-like
        The training HDF5's `ys_sorted` target array, sliced to the
        N rows in use. 0.0 means quenched.

    Returns
    -------
    dict
        {galaxy_id: ssfr}, one float per galaxy.
    """
    values = np.asarray(ys_sorted).reshape(-1)
    ssfrs = {}
    for galaxy_id, row_idxs in galaxy_index.items():
        rows = values[row_idxs]
        if not np.allclose(rows, rows[0]):
            raise ValueError(
                'Rows for {0} disagree on ssfr: {1}'.format(
                    galaxy_id, rows
                )
            )
        ssfrs[galaxy_id] = float(rows[0])
    return ssfrs


def _quantile_edges(values, n_bins):
    """
    Compute interior quantile edges splitting values into n_bins
    bins, collapsing n_bins toward len(values) when there isn't
    enough data for the requested bin count.

    Parameters
    ----------
    values : array-like of float
        Values to bin.
    n_bins : int
        Requested number of bins.

    Returns
    -------
    numpy.ndarray
        n_bins - 1 interior edges (fewer if _quantile_edges
        collapsed n_bins), empty if only one bin fits.
    """
    n_bins = max(1, min(n_bins, len(values)))
    if n_bins <= 1:
        return np.array([])
    quantiles = np.linspace(0., 1., n_bins + 1)[1:-1]
    return np.quantile(values, quantiles)


def _bin_index(value, edges):
    """
    Look up which bin value falls in given interior edges.

    Parameters
    ----------
    value : float
        Value to place.
    edges : numpy.ndarray
        Interior bin edges, from _quantile_edges.

    Returns
    -------
    int
        Bin index, 0 to len(edges).
    """
    return int(np.searchsorted(edges, value, side='right'))


def stratify_galaxies(
        galaxy_ids,
        masses,
        ssfrs,
        n_mass_bins=5,
        n_ssfr_bins=5):
    """
    Group galaxies into strata for representative test-lock
    sampling. Galaxies missing from `masses` land in their own
    'unknown_mass' stratum; stratify_galaxies never drops them.
    stratify_galaxies bins quenched galaxies (ssfr <= 0) by
    log10(Mstar) alone and bins the rest on a log10(Mstar) x
    log10(ssfr) quantile grid. _quantile_edges collapses the bin
    counts toward the available galaxy count, so stratify_galaxies
    degrades gracefully on small datasets instead of erroring.

    Parameters
    ----------
    galaxy_ids : list of str
        Galaxy ids to stratify.
    masses : dict
        {galaxy_id: Mstar}, from load_avg_sfr_csv. Galaxy ids
        absent here are treated as unknown-mass.
    ssfrs : dict
        {galaxy_id: ssfr}, from galaxy_ssfr. Must cover every id in
        galaxy_ids.
    n_mass_bins : int, optional
        Requested quantile bins on log10(Mstar). Default 5.
    n_ssfr_bins : int, optional
        Requested quantile bins on log10(ssfr) for star-forming
        galaxies. Default 5.

    Returns
    -------
    dict
        {stratum_name: [galaxy_id, ...]}. stratum_name is
        'unknown_mass' for galaxies missing from `masses`,
        'quenched_m<mass_bin>' for quenched galaxies, or
        'sf_m<mass_bin>_s<ssfr_bin>' for star-forming galaxies,
        where <mass_bin>/<ssfr_bin> are integer bin indices from
        _bin_index (0 to n_mass_bins - 1 / n_ssfr_bins - 1, fewer
        if bin counts collapsed).
    """
    strata = collections.defaultdict(list)

    known = [g for g in galaxy_ids if g in masses]
    unknown = [g for g in galaxy_ids if g not in masses]
    strata['unknown_mass'].extend(unknown)

    quenched = [g for g in known if ssfrs[g] <= 0.]
    star_forming = [g for g in known if ssfrs[g] > 0.]

    if quenched:
        log_mass = np.log10([masses[g] for g in quenched])
        mass_edges = _quantile_edges(log_mass, n_mass_bins)
        for galaxy_id, log_m in zip(quenched, log_mass):
            mass_bin = _bin_index(log_m, mass_edges)
            strata['quenched_m{0}'.format(mass_bin)].append(
                galaxy_id
            )

    if star_forming:
        log_mass = np.log10([masses[g] for g in star_forming])
        log_ssfr = np.log10([ssfrs[g] for g in star_forming])
        mass_edges = _quantile_edges(log_mass, n_mass_bins)
        ssfr_edges = _quantile_edges(log_ssfr, n_ssfr_bins)
        for galaxy_id, log_m, log_s in zip(
                star_forming, log_mass, log_ssfr):
            mass_bin = _bin_index(log_m, mass_edges)
            ssfr_bin = _bin_index(log_s, ssfr_edges)
            strata['sf_m{0}_s{1}'.format(mass_bin, ssfr_bin)].append(
                galaxy_id
            )

    return dict(strata)


def select_test_lock_galaxies(
        strata,
        target_fraction,
        already_locked,
        seed):
    """
    Draw new galaxies into the test lock so each stratum's locked
    share approaches target_fraction. Draws only from galaxies not
    already in `already_locked`, and never touches an existing
    entry. select_test_lock_galaxies skips the 'unknown_mass'
    stratum entirely, since those galaxies have no mass to make the
    lock representative with.

    Parameters
    ----------
    strata : dict
        {stratum_name: [galaxy_id, ...]}, from stratify_galaxies.
    target_fraction : float
        Target locked share of each stratum, e.g. 0.10.
    already_locked : list of str
        Galaxy ids already in the test lock.
    seed : int
        Seed for the per-stratum random sample.

    Returns
    -------
    list of str
        Newly selected galaxy ids, sorted. Not the full locked set:
        callers combine the returned ids with `already_locked`
        themselves.
    """
    rng = random.Random(seed)
    already_locked = set(already_locked)
    selected = []
    for stratum, galaxy_ids in sorted(strata.items()):
        if stratum == 'unknown_mass':
            continue
        candidates = sorted(
            g for g in galaxy_ids if g not in already_locked
        )
        n_already = len(galaxy_ids) - len(candidates)
        n_target = round(len(galaxy_ids) * target_fraction)
        n_new = min(max(0, n_target - n_already), len(candidates))
        selected.extend(rng.sample(candidates, n_new))
    return sorted(selected)


def _test_lock_version(path):
    """
    Parse the integer version out of a test_lock_v<N>.json path.

    Parameters
    ----------
    path : str or pathlib.Path
        A path whose filename matches 'test_lock_v<N>.json'.

    Returns
    -------
    int
        The version N.
    """
    stem = pathlib.Path(path).stem
    return int(stem.rsplit('_v', 1)[1])


def latest_test_lock_path():
    """
    Find the highest-versioned test lock file in SPLITS_DIR.

    Returns
    -------
    pathlib.Path or None
        The highest-versioned test_lock_v<N>.json in SPLITS_DIR, or
        None if SPLITS_DIR has no test lock files yet.
    """
    candidates = sorted(
        SPLITS_DIR.glob('test_lock_v*.json'),
        key=_test_lock_version,
    )
    return candidates[-1] if candidates else None


def update_test_lock(
        dataset_fname,
        target_fraction=0.10,
        seed=42,
        n_mass_bins=5,
        n_ssfr_bins=5,
        avg_sfr_csv=AVG_SFR_CSV):
    """
    Write the first test lock version in SPLITS_DIR if none exists
    yet, or write a new, higher-versioned test lock that tops up the
    highest existing version. update_test_lock never edits an
    existing test_lock_v<N>.json; it only ever adds galaxies to
    locked_galaxies, and does so by writing the next version's file
    from scratch. If a top-up finds no new galaxies to add,
    update_test_lock writes nothing and returns the existing latest
    version unchanged, so reruns that find the lock already at
    target stay a no-op rather than piling up empty versions.
    update_test_lock calls dataset_lock.verify_dataset(dataset_fname)
    before doing anything else, so it raises immediately if
    dataset_fname has no recorded hash lock yet or if its content
    has drifted since locking.

    Parameters
    ----------
    dataset_fname : str
        Training HDF5 filename, as passed to
        preprocessing.load_metadata (resolved against
        project_data_dir). Must already be locked via
        gallearn.dataset_lock.lock_dataset (e.g. via
        scripts/lock_dataset.py).
    target_fraction : float, optional
        Target locked share of each mass/sSFR stratum. Default
        0.10.
    seed : int, optional
        Seed for the per-stratum random sample. Default 42.
    n_mass_bins : int, optional
        Requested quantile bins on log10(Mstar), passed to
        stratify_galaxies. Default 5.
    n_ssfr_bins : int, optional
        Requested quantile bins on log10(ssfr), passed to
        stratify_galaxies. Default 5.
    avg_sfr_csv : str or pathlib.Path, optional
        Path to the avg_sfrs CSV, passed to load_avg_sfr_csv.
        Defaults to AVG_SFR_CSV.

    Returns
    -------
    tuple of (pathlib.Path, dict)
        The path update_test_lock wrote (or the existing latest
        path, if it added nothing new) and the test lock contents at
        that path (see the module docstring for the schema).
    """
    dataset_lock.verify_dataset(dataset_fname)

    existing_path = latest_test_lock_path()
    existing = _load_json(existing_path) if existing_path else None

    d, N, _ = preprocessing.load_metadata(dataset_fname)
    galaxy_index = build_galaxy_index(d['obs_sorted'][:N])
    ssfrs = galaxy_ssfr(galaxy_index, d['ys_sorted'][:N])
    masses = load_avg_sfr_csv(avg_sfr_csv)

    already_locked = existing['locked_galaxies'] if existing else []

    strata = stratify_galaxies(
        sorted(galaxy_index),
        masses,
        ssfrs,
        n_mass_bins=n_mass_bins,
        n_ssfr_bins=n_ssfr_bins,
    )
    new_galaxies = select_test_lock_galaxies(
        strata, target_fraction, already_locked, seed
    )

    if not new_galaxies and existing is not None:
        print(
            'Test lock at {0} is already at target; nothing to'
            ' add.'.format(existing_path)
        )
        return existing_path, existing

    locked_galaxies = sorted(set(already_locked) | set(new_galaxies))
    n_total = len(galaxy_index)
    updated_at = _now_iso()
    history = list(existing['metadata']['history']) if existing else []
    history.append({
        'updated_at': updated_at,
        'added_count': len(new_galaxies),
        'total_locked': len(locked_galaxies),
    })

    lock_version = (
        _test_lock_version(existing_path) + 1 if existing_path
        else 1
    )
    lock_dict = {
        'lock_version': lock_version,
        'created_at': (
            existing['created_at'] if existing else updated_at
        ),
        'updated_at': updated_at,
        'locked_galaxies': locked_galaxies,
        'metadata': {
            'dataset_path': str(dataset_fname),
            'target_test_fraction': target_fraction,
            'actual_locked_fraction': (
                len(locked_galaxies) / n_total if n_total else 0.
            ),
            'seed': seed,
            'stratification': {
                'n_mass_bins': n_mass_bins,
                'n_ssfr_bins': n_ssfr_bins,
                'bin_allocations': {
                    stratum: len(ids)
                    for stratum, ids in strata.items()
                },
            },
            'history': history,
        },
    }
    output_path = (
        SPLITS_DIR / 'test_lock_v{0}.json'.format(lock_version)
    )
    _save_json(lock_dict, output_path)
    print(
        'Locked {0} new galaxies ({1} total, {2:.1%} of {3}'
        ' galaxies). Wrote {4}.'.format(
            len(new_galaxies),
            len(locked_galaxies),
            lock_dict['metadata']['actual_locked_fraction'],
            n_total,
            output_path,
        )
    )
    return output_path, lock_dict


def create_split_file(
        galaxy_index,
        locked_galaxies,
        val_fraction,
        seed):
    """
    Randomly split the galaxies in galaxy_index that are not in
    locked_galaxies into train and val.

    Parameters
    ----------
    galaxy_index : dict
        {galaxy_id: [row_idx, ...]}, from build_galaxy_index.
    locked_galaxies : list of str
        Galaxy ids the test lock excludes from this split.
    val_fraction : float
        Target share of the non-locked galaxies to put in val.
    seed : int
        Seed for the random shuffle.

    Returns
    -------
    dict
        A split file dict with 'train_galaxies', 'val_galaxies',
        and a 'metadata' dict covering val_fraction/seed/counts.
        write_split adds the dataset_path/test_lock_path/
        test_lock_version keys to that metadata before writing the
        file to disk; create_split_file itself doesn't know those.
    """
    locked = set(locked_galaxies)
    available = sorted(g for g in galaxy_index if g not in locked)
    shuffled = available[:]
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    val_galaxies = sorted(shuffled[:n_val])
    train_galaxies = sorted(shuffled[n_val:])
    return {
        'created_at': _now_iso(),
        'train_galaxies': train_galaxies,
        'val_galaxies': val_galaxies,
        'metadata': {
            'val_fraction': val_fraction,
            'seed': seed,
            'train_galaxy_count': len(train_galaxies),
            'val_galaxy_count': len(val_galaxies),
        },
    }


def _default_split_name(dataset_fname):
    """
    Build a default split_name of '<dataset stem>_v<N>', where N is
    one more than the highest existing split version already written
    for dataset_fname in SPLITS_DIR, or 1 if none exist yet.

    Parameters
    ----------
    dataset_fname : str
        Training HDF5 filename, as passed to write_split.

    Returns
    -------
    str
        '<dataset stem>_v<N>'.
    """
    stem = pathlib.Path(dataset_fname).stem
    versions = []
    for path in SPLITS_DIR.glob('split_{0}_v*.json'.format(stem)):
        try:
            versions.append(int(path.stem.rsplit('_v', 1)[1]))
        except ValueError:
            continue
    return '{0}_v{1}'.format(stem, max(versions, default=0) + 1)


def write_split(
        dataset_fname,
        test_lock_path=None,
        split_name=None,
        val_fraction=0.15,
        seed=42):
    """
    Build a train/val split from the galaxies not in the test lock
    at test_lock_path, and write it to
    SPLITS_DIR / 'split_<split_name>.json'. write_split calls
    dataset_lock.verify_dataset(dataset_fname) before doing anything
    else, so it raises immediately if dataset_fname has no recorded
    hash lock yet or if its content has drifted since locking. Like
    SPLITS_DIR itself, the split's location and naming convention
    aren't caller-configurable, only split_name is: keeping every
    split discoverable in one place matters more than letting a
    caller scatter them under arbitrary paths.

    Parameters
    ----------
    dataset_fname : str
        Training HDF5 filename, as passed to
        preprocessing.load_metadata (resolved against
        project_data_dir). Must already be locked via
        gallearn.dataset_lock.lock_dataset (e.g. via
        scripts/lock_dataset.py).
    test_lock_path : str or pathlib.Path, optional
        Path to a specific test_lock_v<N>.json to split against.
        Defaults to latest_test_lock_path(), the highest version
        currently in SPLITS_DIR. Pass an older version explicitly to
        pin a split to that version instead of whatever is newest.
    split_name : str, optional
        Used in the output filename. Defaults to
        '<dataset_fname's stem>_v<N>' (see _default_split_name), N
        one more than the highest existing split version already
        written for this dataset.
    val_fraction : float, optional
        Target share of the non-locked galaxies to put in val.
        Default 0.15.
    seed : int, optional
        Seed for the random shuffle. Default 42.

    Returns
    -------
    dict
        The split file contents, in the same shape written to disk.
    """
    if test_lock_path is None:
        test_lock_path = latest_test_lock_path()
        if test_lock_path is None:
            raise ValueError(
                'No test lock exists yet in {0}. Run'
                ' `scripts/split.py test-lock`'
                ' first.'.format(SPLITS_DIR)
            )
    test_lock_path = pathlib.Path(test_lock_path)
    test_lock = _load_json(test_lock_path)
    if test_lock is None:
        raise ValueError(
            'No test lock found at {0}. Run `scripts/split.py'
            ' test-lock` first.'.format(test_lock_path)
        )
    print('Using test lock at {0}.'.format(test_lock_path))

    dataset_lock.verify_dataset(dataset_fname)

    d, N, _ = preprocessing.load_metadata(dataset_fname)
    galaxy_index = build_galaxy_index(d['obs_sorted'][:N])

    split_dict = create_split_file(
        galaxy_index,
        test_lock['locked_galaxies'],
        val_fraction,
        seed,
    )
    split_dict['metadata']['dataset_path'] = str(dataset_fname)
    split_dict['metadata']['test_lock_path'] = str(test_lock_path)
    split_dict['metadata']['test_lock_version'] = (
        test_lock['lock_version']
    )

    suffix = split_name if split_name else _default_split_name(
        dataset_fname
    )
    output_path = SPLITS_DIR / 'split_{0}.json'.format(suffix)
    if output_path.exists():
        raise FileExistsError(
            'A split file already exists at {0}. Splits are'
            ' immutable once written, since a checkpoint or a'
            ' resumed run may already be referencing exactly what'
            ' is there; pass a different split_name instead of'
            ' overwriting this one.'.format(output_path)
        )
    _save_json(split_dict, output_path)
    print(
        'Wrote split with {0} train / {1} val galaxies to'
        ' {2}.'.format(
            split_dict['metadata']['train_galaxy_count'],
            split_dict['metadata']['val_galaxy_count'],
            output_path,
        )
    )
    return split_dict


def resolve_split_indices(split_dict, galaxy_index):
    """
    Convert a split's train/val galaxy lists into row-index tensors
    against galaxy_index. resolve_split_indices raises ValueError
    naming any split galaxy absent from galaxy_index, since that
    means the split file no longer matches the dataset galaxy_index
    was built from.

    Parameters
    ----------
    split_dict : dict
        A split file's contents (e.g. from write_split or a loaded
        split JSON), with 'train_galaxies' and 'val_galaxies' keys.
    galaxy_index : dict
        {galaxy_id: [row_idx, ...]}, from build_galaxy_index on the
        HDF5 currently open for training.

    Returns
    -------
    tuple of (torch.LongTensor, torch.LongTensor)
        (train_idxs, val_idxs): row indices into the current HDF5.
    """
    def _rows_for(galaxy_ids):
        missing = [g for g in galaxy_ids if g not in galaxy_index]
        if missing:
            raise ValueError(
                'Split references {0} galaxies not present in the'
                ' current dataset, e.g. {1}'.format(
                    len(missing), missing[:10]
                )
            )
        row_idxs = []
        for galaxy_id in galaxy_ids:
            row_idxs.extend(galaxy_index[galaxy_id])
        return torch.LongTensor(sorted(row_idxs))

    train_idxs = _rows_for(split_dict['train_galaxies'])
    val_idxs = _rows_for(split_dict['val_galaxies'])
    return train_idxs, val_idxs
