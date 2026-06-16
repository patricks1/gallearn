import multiprocessing
import os
import pathlib
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.optimize

from . import config

OCTANT_PROJECTIONS = [
    'projection_ppp',
    'projection_ppm',
    'projection_pmp',
    'projection_pmm',
    'projection_mpp',
    'projection_mpm',
    'projection_mmp',
    'projection_mmm',
]


def _process_galaxy(args):
    '''
    Fit a Sersic2D profile to the r-band image of each octant projection for
    one galaxy. Pool workers accept only a single argument, so the caller
    packs all inputs into a tuple.

    Parameters
    ----------
    args: tuple
        A three-element tuple (gal_id, hdf5_path, queue):
        - gal_id (int): FIREBox galaxy ID extracted from the HDF5 filename.
        - hdf5_path (str): Absolute path to the octant image HDF5 file.
        - queue (multiprocessing.managers.Queue): Shared queue used to report
          per-view progress back to the main process without a shared counter.

    Returns
    -------
    rows: list of dict
        One dict per octant projection (8 total). Each dict contains:
        galaxyID, FOV, pixel, view, band, b_a, PA, n, Re, Ie.
        Failed fits write NaN for b_a, PA, n, Re, and Ie so that downstream
        code can filter them out rather than silently consuming NaN values.
    '''
    import mockobservation_tools.sersic_tools as sersic_tools
    gal_id, hdf5_path, queue = args
    queue.put(('start', gal_id, None))
    rows = []
    with h5py.File(hdf5_path, 'r') as f:
        # FOV and pixel count are uniform across all 8 projections for a
        # given galaxy, so reading from the first group is sufficient.
        fov = f[OCTANT_PROJECTIONS[0]].attrs['FOV']
        pixels = f[OCTANT_PROJECTIONS[0]].attrs['pixels']
        for proj_name in OCTANT_PROJECTIONS:
            band_r = f[proj_name]['band_r'][()]
            try:
                # Promote OptimizeWarning to an exception so that fits that
                # converge to a degenerate solution fall into the except
                # branch rather than returning garbage popt values.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'error',
                        category=scipy.optimize.OptimizeWarning,
                    )
                    popt, _ = sersic_tools.fit_sersic(
                        image=band_r,
                        FOV=fov,
                        sersic_type='sersic2D',
                    )
                # popt order: [amplitude, r_eff, n, x_0, y_0, ellip, theta]
                row = {
                    'galaxyID': gal_id,
                    'FOV': fov,
                    'pixel': pixels,
                    'view': proj_name,
                    'band': 'band_r',
                    'b_a': 1 - popt[5],      # axis ratio = 1 - ellipticity
                    'PA': popt[6] % np.pi,    # position angle in [0, pi)
                    'n': popt[2],
                    'Re': popt[1],
                    'Ie': popt[0],
                }
            except (RuntimeError, scipy.optimize.OptimizeWarning):
                # Write NaN fit columns so the row still exists in the CSV
                # and can be identified as a failed fit.
                row = {
                    'galaxyID': gal_id,
                    'FOV': fov,
                    'pixel': pixels,
                    'view': proj_name,
                    'band': 'band_r',
                    'b_a': np.nan,
                    'PA': np.nan,
                    'n': np.nan,
                    'Re': np.nan,
                    'Ie': np.nan,
                }
            rows.append(row)
            # Notify the main process that one more projection is done so it
            # can update the per-galaxy progress bar.
            queue.put(('view', gal_id, len(rows)))
    queue.put(('done', gal_id, None))
    return rows


def _append_rows_to_csv(rows, path):
    '''
    Append a list of row dicts to a CSV file, writing the header only when
    the file does not yet exist.

    Parameters
    ----------
    rows: list of dict
        Row dicts as returned by _process_galaxy. All dicts must have the
        same keys; pd.DataFrame infers column order from the first dict.
    path: str or pathlib.Path
        Destination CSV path. Created on first call; subsequent calls append
        without repeating the header row.

    Returns
    -------
    None
    '''
    df = pd.DataFrame(rows)
    df.to_csv(
        path,
        mode='a',
        header=not os.path.exists(path),  # write header only for new file
        index=False,
    )


def gen(resume: bool = False):
    '''
    Fit Sersic2D profiles to the octant projections of all galaxies in
    octant_img_dir and write the results to a CSV in project_data_dir. Each
    galaxy's rows are written to disk as soon as its worker finishes, so a
    partial CSV survives an interrupted run.

    Only galaxies whose ID appears in host_2d_shapes or sat_2d_shapes are
    processed. Galaxies absent from both files have no standard-axis fit and
    are not useful for training.

    Output file naming follows these rules:
    - If no output file exists yet, writes to
      AstroPhot_octant_allgals_bandr_Sersic.csv.
    - If that file exists and resume is False, writes to the next available
      versioned name (_v2, _v3, ...) so prior results are never overwritten.
    - If resume is True, appends to the highest-versioned existing file and
      skips galaxies whose galaxyID already appears in it.

    Parameters
    ----------
    resume: bool
        When True, find the highest-versioned existing output CSV, skip
        galaxies whose galaxyID already appears in it, and append new rows to
        that same file. When False (default), start a new versioned file if
        any output file already exists.

    Returns
    -------
    None
    '''
    octant_img_dir = config.config[
        f'{__package__}_paths'
    ]['octant_img_dir']
    project_data_dir = config.config[
        f'{__package__}_paths'
    ]['project_data_dir']

    # Find the highest-versioned existing output file, if any. The base file
    # (no version suffix) counts as v1. This determines both the resume target
    # and the starting point for the next version number.
    out_stem = 'AstroPhot_octant_allgals_bandr_Sersic'
    ver_re = re.compile(
        r'^' + re.escape(out_stem) + r'(?:_v(\d+))?\.csv$'
    )
    versions = []
    for name in os.listdir(project_data_dir):
        m = ver_re.match(name)
        if m:
            v = int(m.group(1)) if m.group(1) else 1
            versions.append((v, pathlib.Path(project_data_dir) / name))
    if versions:
        latest_v, out_path = max(versions, key=lambda x: x[0])
    else:
        latest_v = 0
        out_path = pathlib.Path(project_data_dir) / f'{out_stem}.csv'

    # In non-resume mode, bump to the next version so we never overwrite an
    # existing file. In resume mode we keep out_path pointing at the latest
    # file so we append to it.
    if not resume and out_path.exists():
        next_v = latest_v + 1
        suffix = f'_v{next_v}'
        out_path = pathlib.Path(project_data_dir) / f'{out_stem}{suffix}.csv'
        print(f'Existing file found; writing new results to {out_path.name}')

    # Discover all octant image HDF5 files and extract galaxy IDs from their
    # filenames (pattern: object_<galaxyID>_*.hdf5).
    hdf5_files = sorted([
        f for f in os.listdir(octant_img_dir)
        if f.endswith('.hdf5') or f.endswith('.h5')
    ])

    # The manager queue is shared by all workers so they can send progress
    # messages back to the main process. It must be created before worker_args
    # because each tuple embeds a reference to this queue.
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    worker_args = [
        (int(fname.split('_')[1].split('.')[0]),
         os.path.join(octant_img_dir, fname),
         queue)
        for fname in hdf5_files
    ]

    # Load the galaxy IDs that have standard-axis (non-octant) Sersic fits.
    # We only run octant fitting for these galaxies because get_radii in
    # cnn.py needs the paired standard-axis Re to be meaningful.
    host_ids = pd.read_csv(
        config.config[f'{__package__}_paths']['host_2d_shapes'],
        usecols=['galaxyID'],
    )['galaxyID']
    sat_ids = pd.read_csv(
        config.config[f'{__package__}_paths']['sat_2d_shapes'],
        usecols=['galaxyID'],
    )['galaxyID']
    shape_ids = set(host_ids).union(set(sat_ids))
    n_before_shape = len(worker_args)
    worker_args = [a for a in worker_args if a[0] in shape_ids]
    n_excluded = n_before_shape - len(worker_args)
    if n_excluded:
        print(
            f'{n_excluded} galaxies excluded: not in host_2d_shapes'
            f' or sat_2d_shapes.'
        )

    # In resume mode, skip galaxies that are already present in the output
    # file. A galaxy is considered done if any row with its galaxyID exists,
    # since workers write all 8 projections atomically before signalling done.
    if resume and out_path.exists():
        existing = pd.read_csv(out_path, usecols=['galaxyID'])
        done_ids = set(existing['galaxyID'])
        n_before_resume = len(worker_args)
        worker_args = [a for a in worker_args if a[0] not in done_ids]
        n_skipped = n_before_resume - len(worker_args)
        print(
            f'Resume mode: skipping {n_skipped} completed galaxies,'
            f' {len(worker_args)} remaining.'
        )

    if not worker_args:
        print('No galaxies to process.')
        return

    import rich.live
    import rich.progress

    class RightMofNColumn(rich.progress.ProgressColumn):
        '''Like MofNCompleteColumn but right-justified so fractions with
        different total widths align at the slash.

        Parameters
        ----------
        table_column: rich.table.Column, optional
            Passed through to ProgressColumn.__init__.

        Returns
        -------
        rich.progress.Text
            Right-justified "completed/total" string.
        '''
        def render(self, task):
            total = int(task.total) if task.total else 0
            completed = int(task.completed)
            total_width = len(str(total))
            return rich.progress.Text(
                f'{completed:{total_width}d}/{total}',
                style='progress.download',
                justify='right',
            )

    class LinearETAColumn(rich.progress.ProgressColumn):
        '''ETA computed as (elapsed / fraction_done) * fraction_remaining.
        Unlike the default sliding-window ETA, this never disappears between
        slow task completions.

        Parameters
        ----------
        table_column: rich.table.Column, optional
            Passed through to ProgressColumn.__init__.

        Returns
        -------
        rich.progress.Text
            "eta H:MM:SS" string, or "eta -:--:--" when no data yet.
        '''
        def render(self, task):
            if not task.total or not task.completed:
                return rich.progress.Text(
                    'eta -:--:--', style='progress.remaining',
                )
            elapsed = task.elapsed or 0.0
            remaining = elapsed / task.completed * (
                task.total - task.completed
            )
            hours, rem = divmod(int(remaining), 3600)
            mins, secs = divmod(rem, 60)
            return rich.progress.Text(
                f'eta {hours}:{mins:02d}:{secs:02d}',
                style='progress.remaining',
            )

    # n_galaxies is computed after both filters so the progress bar shows the
    # actual number of galaxies this run will process.
    n_galaxies = len(worker_args)
    n_workers = multiprocessing.cpu_count()

    progress = rich.progress.Progress(
        rich.progress.TextColumn(
            '{task.description}', style='bold',
        ),
        rich.progress.BarColumn(bar_width=30),
        RightMofNColumn(),
        rich.progress.TimeElapsedColumn(),
        LinearETAColumn(),
    )
    overall = progress.add_task(
        f'[cyan]Sersic fits ({n_workers} workers)',
        total=n_galaxies,
    )
    # Maps gal_id to the rich task ID for its per-galaxy progress bar so we
    # can update and remove it when the worker sends view/done messages.
    active_tasks = {}

    n_total = len(worker_args)
    n_completed = 0   # galaxies whose done message has been processed
    n_attempted = 0   # total galaxy-view pairs processed (8 per galaxy)
    n_success = 0
    n_failed = 0
    failed_pairs = []  # (galaxyID, view) tuples for the summary printout

    with (
        multiprocessing.Pool(n_workers) as pool,
        rich.live.Live(progress, refresh_per_second=12),
    ):
        # Dispatch one apply_async call per galaxy so we can collect and write
        # each galaxy's rows as soon as it finishes, rather than waiting for
        # all galaxies to complete as map_async would require.
        async_results = {
            args[0]: pool.apply_async(_process_galaxy, (args,))
            for args in worker_args
        }
        # Keep looping until every galaxy has sent its done message AND the
        # queue is empty. The queue.empty() guard drains any view messages
        # that arrived after the last done was processed.
        while n_completed < n_total or not queue.empty():
            try:
                kind, gal_id, payload = queue.get(timeout=0.1)
            except Exception:
                # queue.get timed out; loop and check the exit condition again.
                continue
            if kind == 'start':
                # Worker just started this galaxy; create its progress bar.
                tid = progress.add_task(
                    f'  galaxy {gal_id}',
                    total=len(OCTANT_PROJECTIONS),
                )
                active_tasks[gal_id] = tid
            elif kind == 'view':
                # Worker finished one more projection; advance the bar.
                tid = active_tasks.get(gal_id)
                if tid is not None:
                    progress.update(tid, completed=payload)
            elif kind == 'done':
                # Worker finished all projections. Collect its rows, write
                # them to disk immediately, then retire the progress bar.
                tid = active_tasks.pop(gal_id, None)
                if tid is not None:
                    progress.remove_task(tid)
                rows = async_results[gal_id].get()
                _append_rows_to_csv(rows, out_path)
                for row in rows:
                    n_attempted += 1
                    if pd.notna(row['Re']):
                        n_success += 1
                    else:
                        n_failed += 1
                        failed_pairs.append((row['galaxyID'], row['view']))
                n_completed += 1
                progress.update(overall, completed=n_completed)

    print(
        f'\n{n_attempted} galaxy-view pairs attempted: '
        f'{n_success} succeeded, {n_failed} failed.'
    )
    if n_failed > 0:
        for gal_id, view in failed_pairs:
            print(f'  object_{gal_id} {view}')
    print(f'\nWrote {out_path}')
