"""
Generate mock galaxy images at 8 viewing angles, one per Cartesian octant.
Each direction is the body diagonal of its octant: (+/-1, +/-1, +/-1)/sqrt(3).
These directions are maximally distinct from one another and from the existing
xy/yz/zx projections. The code saves the images in aug_angles_image_dir
(read from config_<env_name>.ini).

Strategy
--------
mockobservation_tools.galaxy_tools.get_mock_observation only accepts view='xy',
'yz', or 'zx'.  To get an arbitrary line-of-sight n, we rotate all particle
coordinates by a matrix R satisfying R @ n = z_hat and then call the function
with view='xy'.  The xy-projection of the rotated coordinates is equivalent to
viewing along direction n.

Output format
-------------
Per-galaxy HDF5 files matching Courtney's format (as read by
src/image_loader.jl).  Each file contains eight projection groups:

  projection_ppp/
    band_u   float32 (pixels, pixels)
    band_g   float32 (pixels, pixels)
    band_r   float32 (pixels, pixels)
    attrs: copied from source image file, with projection overwritten
  projection_ppm/
    ...
  (one group per octant)

Octant label convention: 'p' = +1, 'm' = -1, in x-y-z order.
E.g. 'pmm' = viewing direction (+1,-1,-1)/sqrt(3).

Galaxy catalogue
----------------
The set of galaxies to process is determined by scanning host_image_dir and
sat_image_dir (read from config_<env_name>.ini) for existing per-galaxy HDF5
files.  The FOV and pixel count are read from each file's attributes.

Simulation units
----------------
FIREbox particle files store masses in units of 1e10 M_sun and positions in
comoving kpc/h.  load_sim_General with the default mass_unit='simulation' and
length_unit='simulation' converts these to physical units (M_sun and physical
kpc) before returning.  This matches the calling convention Courtney uses in
the mockobservation tutorials for all FIREbox data.

Usage
-----
  python scripts/gen_octant_images.py
"""

import contextlib
import io
import multiprocessing
import pathlib

import h5py
import numpy as np

import mockobservation_tools.galaxy_tools as gt


# Body-diagonal unit vectors, one per octant.  The label encodes the sign of
# each Cartesian component: p = +1, m = -1.
OCTANT_DIRECTIONS = {
    'ppp': np.array([ 1.,  1.,  1.]) / np.sqrt(3.),
    'ppm': np.array([ 1.,  1., -1.]) / np.sqrt(3.),
    'pmp': np.array([ 1., -1.,  1.]) / np.sqrt(3.),
    'pmm': np.array([ 1., -1., -1.]) / np.sqrt(3.),
    'mpp': np.array([-1.,  1.,  1.]) / np.sqrt(3.),
    'mpm': np.array([-1.,  1., -1.]) / np.sqrt(3.),
    'mmp': np.array([-1., -1.,  1.]) / np.sqrt(3.),
    'mmm': np.array([-1., -1., -1.]) / np.sqrt(3.),
}

OCTANT_LABELS = list(OCTANT_DIRECTIONS.keys())


def rotation_matrix_to_z(n):
    """
    3x3 rotation matrix R such that R @ n == [0, 0, 1].  Uses Rodrigues'
    rotation formula; the degenerate case n == -z_hat is handled with a
    180-degree rotation about the x-axis.
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0., 0., 1.])
    v = np.cross(n, z)
    c = float(np.dot(n, z))
    s = float(np.linalg.norm(v))
    if s < 1e-10:
        if c > 0.:
            return np.eye(3)
        # n is antiparallel to z: rotate 180 degrees around the x-axis
        return np.array(
            [[ 1.,  0.,  0.],
             [ 0., -1.,  0.],
             [ 0.,  0., -1.]]
        )
    k = v / s
    K = np.array(
        [[ 0.,   -k[2],  k[1]],
         [ k[2],  0.,   -k[0]],
         [-k[1],  k[0],  0.  ]]
    )
    return (
        c * np.eye(3)
        + (1. - c) * np.outer(k, k)
        + s * K
    )


def rotate_snapdict(snapdict, R):
    """
    Shallow copy of snapdict with Coordinates rotated by R.

    The 3D radius 'r' is invariant under rotation and does not need updating.
    get_mock_observation only uses 'r' for the FOV mask, so this shallow copy
    is sufficient.
    """
    d = snapdict.copy()
    # Coordinates has shape (N, 3); the row-vector convention requires R.T
    d['Coordinates'] = snapdict['Coordinates'] @ R.T
    return d


def scan_image_dirs(host_dir, sat_dir):
    """
    Scan host and satellite image directories for existing per-galaxy
    HDF5 files.  Opens each file and reads attributes from the first
    projection group.  Returns a list of dicts with keys: attrs (dict
    of all attributes from the source file), fov, pixels, fname.
    Deduplicates by galaxyID attribute.
    """
    galaxies = []
    seen_ids = set()
    for dirpath in (host_dir, sat_dir):
        d = pathlib.Path(dirpath)
        if not d.exists():
            print(
                f'Warning: {d} does not exist, skipping.'
            )
            continue
        for f in sorted(d.glob('*.hdf5')):
            try:
                with h5py.File(f, 'r') as h:
                    grp_name = list(h.keys())[0]
                    attrs = dict(h[grp_name].attrs)
            except Exception:
                continue
            gal_id = attrs.get('galaxyID')
            if gal_id is None or gal_id in seen_ids:
                continue
            seen_ids.add(gal_id)
            galaxies.append({
                'attrs': attrs,
                'fov': int(attrs['FOV']),
                'pixels': int(attrs['pixels']),
                'fname': f.name,
            })
    return galaxies


def process_galaxy(gal, objects_dir, output_dir, queue):
    """Generate octant images for a single galaxy.  Sends
    progress updates to queue and returns a result dict."""
    src_attrs = gal['attrs']
    gal_id = int(src_attrs['galaxyID'])
    fov = gal['fov']
    pixels = gal['pixels']
    fname = gal['fname']

    obj_path = (
        objects_dir
        / f'particles_within_Rvir_object_{gal_id}.hdf5'
    )
    ahf_path = (
        objects_dir
        / f'bound_particle_filters_object_{gal_id}.hdf5'
    )

    if not obj_path.exists():
        queue.put(('skip', gal_id, obj_path.name))
        return {
            'gal_id': gal_id,
            'success': False,
            'message': f'{obj_path.name} not found',
        }

    ahf_arg = (
        str(ahf_path) if ahf_path.exists() else None
    )

    queue.put(('load', gal_id, None))

    # Redirect stdout to suppress verbose printouts from
    # mockobservation_tools and its dependencies (e.g. L/M
    # band calculation messages).
    devnull = io.StringIO()

    try:
        # load_sim_General defaults to mass_unit='simulation'
        # and length_unit='simulation', matching the FIREbox
        # calling convention in Courtney's mockobservation
        # tutorials.
        with contextlib.redirect_stdout(devnull):
            star_sd, gas_sd = gt.load_sim_General(
                str(obj_path),
                ahf_path=ahf_arg,
            )
    except Exception as exc:
        queue.put(('error', gal_id, str(exc)))
        return {
            'gal_id': gal_id,
            'success': False,
            'message': f'load error: {exc}',
        }

    out_path = output_dir / fname
    try:
        with h5py.File(out_path, 'w') as f:
            for oct_i, (label, n) in enumerate(
                    OCTANT_DIRECTIONS.items()):
                R = rotation_matrix_to_z(n)
                star_rot = rotate_snapdict(star_sd, R)
                gas_rot = rotate_snapdict(gas_sd, R)

                with contextlib.redirect_stdout(devnull):
                    band_u, band_g, band_r = (
                        gt.get_mock_observation(
                            star_rot,
                            gas_rot,
                            bands=[1, 2, 3],
                            FOV=fov,
                            pixels=pixels,
                            view='xy',
                            center='none',
                            return_type='SB_lum',
                            QUIET=True,
                        )
                    )

                grp_name = f'projection_{label}'
                grp = f.create_group(grp_name)
                for k, v in src_attrs.items():
                    grp.attrs[k] = v
                grp.attrs['projection'] = label
                grp.create_dataset(
                    'band_u',
                    data=np.float32(band_u),
                )
                grp.create_dataset(
                    'band_g',
                    data=np.float32(band_g),
                )
                grp.create_dataset(
                    'band_r',
                    data=np.float32(band_r),
                )

                queue.put(('octant', gal_id, oct_i + 1))

    except Exception as exc:
        if out_path.exists():
            out_path.unlink()
        queue.put(('error', gal_id, str(exc)))
        return {
            'gal_id': gal_id,
            'success': False,
            'message': f'image generation error: {exc}',
        }

    queue.put(('done', gal_id, None))
    return {
        'gal_id': gal_id,
        'success': True,
        'message': f'saved {out_path.name}',
    }


def _worker(args):
    """Unpack arguments for process_galaxy so it works with
    Pool.imap_unordered."""
    return process_galaxy(*args)


def main():
    import rich.live
    import rich.progress
    import rich.table

    import gallearn

    config_paths = gallearn.config.config['gallearn_paths']

    # host_image_dir and sat_image_dir come exclusively from
    # config_<env_name>.ini so that this script always uses the
    # same directories as image_loader.jl.
    host_image_dir = config_paths['host_image_dir']
    sat_image_dir = config_paths['sat_image_dir']

    firebox_dir = pathlib.Path(
        config_paths['firebox_data_dir']
    )
    output_dir = pathlib.Path(
        config_paths['aug_angles_image_dir']
    )
    objects_dir = firebox_dir / 'objects_1200_original'

    # Scan existing image directories to build the galaxy
    # catalogue.  Each entry carries the source file's
    # attributes, FOV, and pixel count.
    galaxies = scan_image_dirs(host_image_dir, sat_image_dir)
    n_galaxies = len(galaxies)
    if n_galaxies == 0:
        print('No galaxies found. Nothing to do.')
        raise SystemExit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    work_args = [
        (gal, objects_dir, output_dir, queue)
        for gal in galaxies
    ]

    # Set up the rich progress display.  The overall bar
    # tracks completed galaxies; per-galaxy bars show octant
    # progress and appear/disappear as workers pick up tasks.
    progress = rich.progress.Progress(
        rich.progress.TextColumn(
            '{task.description}', style='bold',
        ),
        rich.progress.BarColumn(bar_width=30),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeElapsedColumn(),
    )

    overall_task = progress.add_task(
        f'[cyan]Overall ({n_workers} workers)',
        total=n_galaxies,
    )

    # Maps gal_id -> rich task_id for active galaxy bars.
    active_tasks = {}

    n_done = 0
    n_skip = 0
    n_finished = 0

    with (
        multiprocessing.Pool(n_workers) as pool,
        rich.live.Live(progress, refresh_per_second=12),
    ):
        async_results = pool.starmap_async(
            process_galaxy, work_args,
        )

        while not async_results.ready() or not queue.empty():
            try:
                msg = queue.get(timeout=0.1)
            except Exception:
                continue

            kind, gal_id, payload = msg

            if kind == 'load':
                tid = progress.add_task(
                    f'  galaxy {gal_id}',
                    total=8,
                )
                active_tasks[gal_id] = tid

            elif kind == 'octant':
                tid = active_tasks.get(gal_id)
                if tid is not None:
                    progress.update(
                        tid, completed=payload,
                    )

            elif kind == 'done':
                tid = active_tasks.pop(gal_id, None)
                if tid is not None:
                    progress.update(
                        tid,
                        completed=8,
                        description=(
                            f'  galaxy {gal_id} [green]✓'
                        ),
                    )
                    progress.remove_task(tid)
                n_done += 1
                n_finished += 1
                progress.update(
                    overall_task, completed=n_finished,
                )

            elif kind == 'skip':
                n_skip += 1
                n_finished += 1
                progress.update(
                    overall_task, completed=n_finished,
                )

            elif kind == 'error':
                tid = active_tasks.pop(gal_id, None)
                if tid is not None:
                    progress.update(
                        tid,
                        description=(
                            f'  galaxy {gal_id}'
                            f' [red]✗ {payload}'
                        ),
                    )
                    progress.remove_task(tid)
                n_skip += 1
                n_finished += 1
                progress.update(
                    overall_task, completed=n_finished,
                )

        # Drain any remaining messages.
        while not queue.empty():
            try:
                msg = queue.get_nowait()
                kind, gal_id, payload = msg
                if kind in ('done', 'skip', 'error'):
                    n_finished += 1
                    if kind == 'done':
                        n_done += 1
                    else:
                        n_skip += 1
                    progress.update(
                        overall_task,
                        completed=n_finished,
                    )
                    tid = active_tasks.pop(
                        gal_id, None
                    )
                    if tid is not None:
                        progress.remove_task(tid)
            except Exception:
                break

    print(
        f'\nDone. {n_done} galaxies processed,'
        f' {n_skip} skipped.',
    )


if __name__ == '__main__':
    main()
