import multiprocessing
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.optimize
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

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
    Fit a Sersic2D profile to the r-band of each octant projection for one
    galaxy. Pool.map requires a single argument, so the caller packs all
    inputs into a tuple.

    Parameters
    ----------
    args: tuple
        (gal_id, hdf5_path, queue) where gal_id is the FIREBox galaxy ID,
        hdf5_path is the path to the octant image HDF5 file, and queue is
        a multiprocessing manager Queue used to report per-view progress
        back to the main process.

    Returns
    -------
    rows: list of dict
        One dict per octant projection. Each dict has the columns listed in
        the gen_octant_shapes plan: galaxyID, FOV, pixel, view, band, b_a,
        PA, n, Re, Ie. Failed fits write NaN for all fit columns.
    '''
    import mockobservation_tools.sersic_tools as sersic_tools
    gal_id, hdf5_path, queue = args
    rows = []
    with h5py.File(hdf5_path, 'r') as f:
        # FOV and pixel count are the same across all projections for a
        # given galaxy, so read them from the first group.
        fov = f[OCTANT_PROJECTIONS[0]].attrs['FOV']
        pixels = f[OCTANT_PROJECTIONS[0]].attrs['pixels']
        for proj_name in OCTANT_PROJECTIONS:
            band_r = f[proj_name]['band_r'][()]
            try:
                # Treat OptimizeWarning as an error so failed fits fall
                # through to the except branch instead of returning garbage
                # popt values.
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
                    'b_a': 1 - popt[5],
                    'PA': popt[6] % np.pi,
                    'n': popt[2],
                    'Re': popt[1],
                    'Ie': popt[0],
                }
            except (RuntimeError, scipy.optimize.OptimizeWarning):
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
            queue.put(1)
    return rows


def gen():
    '''
    Fit Sersic2D profiles to the octant projections of all galaxies in
    octant_img_dir and write the results to
    project_data_dir/AstroPhot_octant_allgals_bandr_Sersic.csv.

    The output CSV has the same columns as the existing host and satellite
    shape files (host_2d_shapes, sat_2d_shapes) so that get_radii in cnn.py
    can concatenate all three and look up Re uniformly. Galaxy-view pairs
    whose fits fail are written with NaN fit columns so get_radii can drop
    them rather than silently including them with NaN Re.

    Call as:
        python -c "import gallearn.gen_octant_shapes;
                   gallearn.gen_octant_shapes.gen()"
    '''
    octant_img_dir = config.config[
        f'{__package__}_paths'
    ]['octant_img_dir']
    project_data_dir = config.config[
        f'{__package__}_paths'
    ]['project_data_dir']

    hdf5_files = sorted([
        f for f in os.listdir(octant_img_dir)
        if f.endswith('.hdf5') or f.endswith('.h5')
    ])
    # Each worker needs the manager queue so it can report per-view progress
    # back to the main process without a shared counter.
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    worker_args = [
        (int(fname.split('_')[1].split('.')[0]),
         os.path.join(octant_img_dir, fname),
         queue)
        for fname in hdf5_files
    ]

    n_views = len(worker_args) * len(OCTANT_PROJECTIONS)

    all_rows = []
    with Progress(
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        overall = progress.add_task(
            'Fitting Sersic profiles', total=n_views
        )
        with multiprocessing.Pool() as pool:
            result = pool.map_async(_process_galaxy, worker_args)
            completed = 0
            while not result.ready():
                while not queue.empty():
                    queue.get_nowait()
                    completed += 1
                    progress.update(overall, completed=completed)
            galaxy_rows_list = result.get()
            # Drain any queue items that arrived after result.ready().
            while not queue.empty():
                queue.get_nowait()
            progress.update(overall, completed=n_views)

    for galaxy_rows in galaxy_rows_list:
        all_rows.extend(galaxy_rows)

    df = pd.DataFrame(all_rows)

    n_attempted = len(df)
    n_success = df['Re'].notna().sum()
    n_failed = n_attempted - n_success
    print(
        f'\n{n_attempted} galaxy-view pairs attempted: '
        f'{n_success} succeeded, {n_failed} failed.'
    )
    if n_failed > 0:
        failed = df[df['Re'].isna()][['galaxyID', 'view']]
        for _, row in failed.iterrows():
            print(f'  object_{row["galaxyID"]} {row["view"]}')

    out_path = os.path.join(
        project_data_dir,
        'AstroPhot_octant_allgals_bandr_Sersic.csv',
    )
    df.to_csv(out_path, index=False)
    print(f'\nWrote {out_path}')
