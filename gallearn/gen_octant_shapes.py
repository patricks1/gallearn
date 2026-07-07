import multiprocessing
import os
import pathlib
import re
import warnings

import h5py
import numpy as np
import pandas as pd
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

CSV_COLUMNS = [
    'galaxyID', 'FOV', 'pixel', 'view', 'band',
    'b_a', 'PA', 'n', 'Re', 'Ie_log10',
]
ASTROPHOT_CSV_COLUMNS = CSV_COLUMNS + ['converged']


def _fit_astrophot(band_r, fov, pixels):
    '''Fit a Sersic profile to band_r using AstroPhot's LM optimizer.

    Parameters
    ----------
    band_r: np.ndarray
        2D r-band image array.
    fov: float
        Field of view in kpc.
    pixels: int
        Number of pixels along one image side.

    Returns
    -------
    tuple
        (b_a, PA, n, Re, Ie, converged) where Re is in kpc, Ie is
        log10(flux/arcsec^2) (matching the units of Courtney's existing
        host/sat shape CSVs, since the installed AstroPhot version reports
        Ie linearly), and converged is the LM optimizer's status message
        (e.g. 'success', 'fail. Maximum iterations').
    '''
    import astrophot as ap
    pixel_scale = 2 * fov / pixels
    target = ap.image.TargetImage(data=band_r, pixelscale=1)
    model = ap.models.SersicGalaxy(name='galaxy', target=target)
    model.initialize()
    result = ap.fit.LM(model, verbose=0).fit()
    b_a = model.q.value.item()
    PA = model.PA.value.item()
    n = model.n.value.item()
    Re = model.Re.value.item() * pixel_scale
    Ie = np.log10(model.Ie.value.item())
    converged = result.message
    return b_a, PA, n, Re, Ie, converged


def _fit_sersic_2d(band_r, fov):
    '''Fit a Sersic2D profile using mockobservation_tools.sersic_tools.

    Parameters
    ----------
    band_r: np.ndarray
        2D r-band image array.
    fov: float
        Field of view in kpc.

    Returns
    -------
    tuple of float
        (b_a, PA, n, Re, Ie) where Re is in kpc and Ie is
        log10(L_sun/kpc^2), matching the log10 convention used for the
        AstroPhot backend.
    '''
    import mockobservation_tools.sersic_tools as sersic_tools
    import scipy.optimize
    # Promote OptimizeWarning to an exception so fits that converge to a
    # degenerate solution raise rather than return garbage popt values.
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
    b_a = 1 - popt[5]        # axis ratio = 1 - ellipticity
    PA = popt[6] % np.pi     # position angle in [0, pi)
    n = popt[2]
    Re = popt[1]
    Ie = np.log10(popt[0])
    return b_a, PA, n, Re, Ie


def _process_galaxy(args):
    '''
    Fit a Sersic profile to the r-band image of each octant projection for
    one galaxy. Pool workers accept only a single argument, so the caller
    packs all inputs into a tuple.

    Each completed projection row is sent through the queue as a
    ('row', gal_id, row_dict) message so the main process can write it to
    disk immediately. Projections listed in skip_views are skipped without
    fitting; only a 'view' progress message is sent for them.

    Parameters
    ----------
    args: tuple
        A five-element tuple (gal_id, hdf5_path, queue, skip_views, fitter):
        - gal_id (int): FIREBox galaxy ID extracted from the HDF5 filename.
        - hdf5_path (str): Absolute path to the octant image HDF5 file.
        - queue (multiprocessing.managers.Queue): Shared queue used to
          report per-projection rows and progress back to the main process.
        - skip_views (frozenset): Projection names already present in the
          output CSV. These projections are skipped; only a 'view' progress
          message is emitted so the progress bar stays consistent.
        - fitter (str): Fitting backend. 'astrophot' calls AstroPhot's LM
          optimizer; 'fit_sersic' calls sersic_tools.fit_sersic from
          mockobservation_tools.

    Returns
    -------
    None
        Rows are delivered via queue messages rather than as a return value.
        Fits that raise an exception send NaN for b_a, PA, n, Re, and
        Ie_log10 so that downstream code can filter them out rather than
        silently consuming NaN values. AstroPhot fits reporting
        non-convergence do not raise and are not NaN'd out; their real
        fitted values are kept, with the non-convergence noted separately
        in the 'converged' column.
    '''
    gal_id, hdf5_path, queue, skip_views, fitter = args
    queue.put(('start', gal_id, len(skip_views)))
    n_sent = 0
    with h5py.File(hdf5_path, 'r') as f:
        # FOV and pixel count are uniform across all 8 projections for a
        # given galaxy, so reading from the first group is sufficient.
        fov = f[OCTANT_PROJECTIONS[0]].attrs['FOV']
        pixels = f[OCTANT_PROJECTIONS[0]].attrs['pixels']
        for proj_name in OCTANT_PROJECTIONS:
            if proj_name in skip_views:
                continue
            band_r = f[proj_name]['band_r'][()]
            try:
                if fitter == 'astrophot':
                    b_a, PA, n, Re, Ie, converged = _fit_astrophot(
                        band_r, fov, pixels,
                    )
                else:
                    b_a, PA, n, Re, Ie = _fit_sersic_2d(band_r, fov)
                row = {
                    'galaxyID': gal_id,
                    'FOV': fov,
                    'pixel': pixels,
                    'view': proj_name,
                    'band': 'band_r',
                    'b_a': b_a,
                    'PA': PA,
                    'n': n,
                    'Re': Re,
                    'Ie_log10': Ie,
                }
                if fitter == 'astrophot':
                    row['converged'] = converged
            except Exception:
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
                    'Ie_log10': np.nan,
                }
                if fitter == 'astrophot':
                    row['converged'] = np.nan
            queue.put(('row', gal_id, row))
            n_sent += 1
            queue.put(('view', gal_id, len(skip_views) + n_sent))
    queue.put(('done', gal_id, None))


def _sort_csv(path):
    '''Read the CSV at path, sort by galaxyID then view, and overwrite it.

    Parameters
    ----------
    path: pathlib.Path
        CSV file to sort in place.

    Returns
    -------
    None
    '''
    df = pd.read_csv(path)
    df = df.sort_values(['galaxyID', 'view']).reset_index(drop=True)
    df.to_csv(path, index=False)


def _append_rows_to_csv(rows, path):
    '''
    Append a list of row dicts to a CSV file. The file must already exist
    with a header row (created by gen() at startup); this function always
    appends without writing a header.

    Parameters
    ----------
    rows: list of dict
        Row dicts as returned by _process_galaxy. All dicts must have the
        same keys; pd.DataFrame infers column order from the first dict.
    path: str or pathlib.Path
        Destination CSV path.

    Returns
    -------
    None
    '''
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=False, index=False)


def _run_fitting_pass(worker_args, out_path, label):
    '''
    Run one fitting pass: dispatch worker_args to a multiprocessing pool,
    collect per-projection rows from the queue, write each row to out_path
    immediately, and display a rich progress bar.

    Parameters
    ----------
    worker_args: list of tuple
        Each tuple is (gal_id, hdf5_path, queue, skip_views, fitter) as
        accepted by _process_galaxy.
    out_path: pathlib.Path
        CSV file to append rows to. Must already exist with a header.
    label: str
        Label shown on the overall progress bar (e.g. 'Sersic fits').

    Returns
    -------
    tuple: (n_attempted, n_success, n_failed, failed_pairs)
    '''
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
        f'[cyan]{label} ({n_workers} workers)',
        total=n_galaxies,
    )
    active_tasks = {}

    n_total = n_galaxies
    n_completed = 0
    n_attempted = 0
    n_success = 0
    n_failed = 0
    failed_pairs = []

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    # Inject the managed queue into each worker arg tuple (replacing the
    # placeholder queue that was set to None before this call).
    worker_args = [
        (gal_id, hdf5_path, queue, skip_views, fitter)
        for gal_id, hdf5_path, _, skip_views, fitter in worker_args
    ]

    with (
        multiprocessing.Pool(n_workers) as pool,
        rich.live.Live(progress, refresh_per_second=12),
    ):
        for args in worker_args:
            pool.apply_async(_process_galaxy, (args,))
        while n_completed < n_total or not queue.empty():
            try:
                kind, gal_id, payload = queue.get(timeout=0.1)
            except Exception:
                continue
            if kind == 'start':
                tid = progress.add_task(
                    f'  galaxy {gal_id}',
                    total=len(OCTANT_PROJECTIONS),
                    completed=payload,
                )
                active_tasks[gal_id] = tid
            elif kind == 'view':
                tid = active_tasks.get(gal_id)
                if tid is not None:
                    progress.update(tid, completed=payload)
            elif kind == 'row':
                _append_rows_to_csv([payload], out_path)
                n_attempted += 1
                if pd.notna(payload['Re']):
                    n_success += 1
                else:
                    n_failed += 1
                    failed_pairs.append((payload['galaxyID'], payload['view']))
            elif kind == 'done':
                tid = active_tasks.pop(gal_id, None)
                if tid is not None:
                    progress.remove_task(tid)
                n_completed += 1
                progress.update(overall, completed=n_completed)

    return n_attempted, n_success, n_failed, failed_pairs


def gen(resume: bool = False, fitter: str = 'astrophot'):
    '''
    Fit Sersic profiles to the octant projections of all galaxies in
    octant_img_dir and write the results to a CSV in project_data_dir. Each
    projection row is written to disk immediately after its fit completes,
    so an interrupted run loses at most the projection currently in flight.

    Only galaxies whose ID appears in host_2d_shapes or sat_2d_shapes are
    processed. Galaxies absent from both files have no standard-axis fit and
    are not useful for training.

    The output CSV is created (header only) at startup before any fitting
    begins. A second concurrent instance of gen() will see the file already
    exists and write to a new versioned file, avoiding collisions.

    After the main fitting loop finishes, gen() checks for any
    (galaxyID, view) pairs that were expected but are absent from the CSV
    and runs a backfill pass to fill them in.

    Output file naming encodes the fitter:
    - fitter='astrophot': AstroPhot_octant_allgals_bandr_Sersic[_vN].csv
    - fitter='fit_sersic': FitSersic_octant_allgals_bandr_Sersic[_vN].csv
    If no versioned file exists, writes to the base name. If one exists and
    resume is False, writes to the next version (_v2, _v3, ...) so prior
    results are never overwritten.

    Parameters
    ----------
    resume: bool
        When True, find the highest-versioned existing output CSV, skip
        (galaxyID, view) pairs already present in it, and append new rows
        to that same file. When False (default), start a new versioned file
        if any output file already exists.
    fitter: str
        Fitting backend. 'astrophot' uses AstroPhot's LM optimizer (default).
        'fit_sersic' uses sersic_tools.fit_sersic from mockobservation_tools.

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
    prefix = 'AstroPhot' if fitter == 'astrophot' else 'FitSersic'
    out_stem = f'{prefix}_octant_allgals_bandr_Sersic'
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
    # existing file. In resume mode keep out_path pointing at the latest file.
    if not resume and out_path.exists():
        next_v = latest_v + 1
        suffix = f'_v{next_v}'
        out_path = pathlib.Path(project_data_dir) / f'{out_stem}{suffix}.csv'
        print(f'Existing file found; writing new results to {out_path.name}')
    else:
        print(f'Output file: {out_path.name}')

    # Create the output CSV (header only) immediately so that a second
    # concurrent instance of gen() sees the file and starts a new version
    # rather than colliding with this one.
    if not out_path.exists():
        columns = (
            ASTROPHOT_CSV_COLUMNS if fitter == 'astrophot' else CSV_COLUMNS
        )
        pd.DataFrame(columns=columns).to_csv(out_path, index=False)

    # Discover all octant image HDF5 files and extract galaxy IDs from their
    # filenames (pattern: object_<galaxyID>_*.hdf5).
    hdf5_files = sorted([
        f for f in os.listdir(octant_img_dir)
        if f.endswith('.hdf5') or f.endswith('.h5')
    ])

    # queue is a placeholder; _run_fitting_pass creates its own managed queue
    # and injects it. We use None here so worker_args tuples have the right
    # shape before the queue exists.
    worker_args = [
        (int(fname.split('_')[1].split('.')[0]),
         os.path.join(octant_img_dir, fname),
         None,
         frozenset(),
         fitter)
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

    # Save the full list before resume filtering so the completeness check
    # after the main pass covers every galaxy this invocation intended to fit.
    original_worker_args = list(worker_args)

    # In resume mode, read existing (galaxyID, view) pairs and build per-
    # galaxy skip sets. Galaxies with all 8 projections present are dropped
    # entirely; partially done galaxies are re-queued with skip_views set to
    # the projections already in the CSV.
    if resume and out_path.exists():
        # Guard against resuming a file that was actually produced by a
        # different fitter than requested here: check both its name and
        # its columns against what this fitter would write.
        if not out_path.name.startswith(f'{prefix}_'):
            raise ValueError(
                f"{out_path.name} does not start with '{prefix}_', which"
                f' fitter={fitter!r} requires. Refusing to resume: this'
                f' file was likely produced by a different fitter.'
            )
        expected_columns = (
            ASTROPHOT_CSV_COLUMNS if fitter == 'astrophot' else CSV_COLUMNS
        )
        existing_columns = pd.read_csv(out_path, nrows=0).columns.tolist()
        if existing_columns != expected_columns:
            raise ValueError(
                f'{out_path.name} has columns {existing_columns}, which'
                f' does not match the columns fitter={fitter!r} would'
                f' write ({expected_columns}). Refusing to resume: this'
                f' file was likely produced by a different fitter.'
            )
        existing = pd.read_csv(out_path, usecols=['galaxyID', 'view'])
        done_pairs = set(zip(existing['galaxyID'], existing['view']))
        new_worker_args = []
        n_skipped = 0
        for gal_id, hdf5_path, q, _, _ in worker_args:
            skip = frozenset(
                v for v in OCTANT_PROJECTIONS
                if (gal_id, v) in done_pairs
            )
            if len(skip) == len(OCTANT_PROJECTIONS):
                n_skipped += 1
            else:
                new_worker_args.append(
                    (gal_id, hdf5_path, q, skip, fitter)
                )
        worker_args = new_worker_args
        n_partial = sum(
            1 for _, _, _, s, _ in worker_args if s
        )
        print(
            f'Resume mode: skipping {n_skipped} completed galaxies,'
            f' {n_partial} partially done, {len(worker_args)} remaining.'
        )
        _sort_csv(out_path)

    if not worker_args:
        print('No galaxies to process.')
        return

    n_attempted, n_success, n_failed, failed_pairs = _run_fitting_pass(
        worker_args,
        out_path,
        label='Sersic fits',
    )
    _sort_csv(out_path)

    # Check that every expected (galaxyID, view) pair landed in the CSV.
    # Missing pairs indicate rows lost to a crash mid-write; backfill them.
    existing = pd.read_csv(out_path, usecols=['galaxyID', 'view'])
    done_pairs = set(zip(existing['galaxyID'], existing['view']))
    backfill_args = []
    for gal_id, hdf5_path, _, _, _ in original_worker_args:
        missing = frozenset(
            v for v in OCTANT_PROJECTIONS
            if (gal_id, v) not in done_pairs
        )
        if missing:
            backfill_args.append(
                (gal_id, hdf5_path, None, missing, fitter)
            )

    if backfill_args:
        print(
            f'\nWarning: {len(backfill_args)} galaxies have missing'
            f' projections after the main pass. Running backfill.'
        )
        for gal_id, _, _, missing, _ in backfill_args:
            for view in sorted(missing):
                print(f'  object_{gal_id} {view}')
        bf_attempted, bf_success, bf_failed, bf_pairs = _run_fitting_pass(
            backfill_args,
            out_path,
            label='Backfill',
        )
        n_attempted += bf_attempted
        n_success += bf_success
        n_failed += bf_failed
        failed_pairs.extend(bf_pairs)
        _sort_csv(out_path)

    print(
        f'\n{n_attempted} galaxy-view pairs attempted: '
        f'{n_success} succeeded, {n_failed} failed.'
    )
    if n_failed > 0:
        for gal_id, view in failed_pairs:
            print(f'  object_{gal_id} {view}')
    print(f'\nWrote {out_path}')
