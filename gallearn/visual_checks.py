def plot_mass_ssfr_distributions(
        dataset_fname,
        test_lock_path=None,
        avg_sfr_csv=None,
        n_bins=30):
    """
    Plot log10(Mstar) and log10(ssfr) histograms comparing the full
    galaxy population, the population excluding the locked test set,
    and the locked test set itself, so a stratified test lock's
    representativeness is visible directly rather than only implied
    by the lock file's recorded bin_allocations counts.

    Excludes galaxies missing from avg_sfr_csv from every panel
    (their mass is unknown). log10 is undefined for a quenched
    galaxy's ssfr (<= 0), so the sSFR panel represents each group's
    quenched galaxies as a single 'Quenched' bar instead of placing
    them on the continuous log10(ssfr) axis, matching how
    stratify_galaxies treats quenched galaxies as their own stratum
    rather than a point on the star-forming mass/sSFR grid. The
    'Quenched' bar and the continuous histogram share one
    normalization per group (both divide by that group's total
    galaxy count, not just its star-forming count), so the bar's
    area and the histogram's area sum to 1 and are directly
    comparable.

    Parameters
    ----------
    dataset_fname : str
        Training HDF5 filename, as passed to
        preprocessing.load_metadata. Must already be locked via
        gallearn.dataset_lock.lock_dataset (e.g. via
        scripts/lock_dataset.py).
    test_lock_path : str or pathlib.Path, optional
        Path to a specific test_lock_v<N>.json. Defaults to
        splitting.latest_test_lock_path(), the highest version
        currently in splitting.SPLITS_DIR.
    avg_sfr_csv : str or pathlib.Path, optional
        Path to the avg_sfrs CSV, passed to
        splitting.load_avg_sfr_csv. Defaults to
        splitting.AVG_SFR_CSV.
    n_bins : int, optional
        Number of histogram bins per panel. Default 30.

    Returns
    -------
    None
    """
    import json

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    from . import dataset_lock
    from . import preprocessing
    from . import splitting

    if avg_sfr_csv is None:
        avg_sfr_csv = splitting.AVG_SFR_CSV

    if test_lock_path is None:
        test_lock_path = splitting.latest_test_lock_path()
        if test_lock_path is None:
            raise ValueError(
                'No test lock exists yet in {0}. Run'
                ' `scripts/split.py test-lock`'
                ' first.'.format(splitting.SPLITS_DIR)
            )
    with open(test_lock_path) as f:
        test_lock = json.load(f)
    locked_galaxies = set(test_lock['locked_galaxies'])

    dataset_lock.verify_dataset(dataset_fname)
    d, N, _ = preprocessing.load_metadata(dataset_fname)
    galaxy_index = splitting.build_galaxy_index(d['obs_sorted'][:N])
    ssfrs = splitting.galaxy_ssfr(galaxy_index, d['ys_sorted'][:N])
    masses = splitting.load_avg_sfr_csv(avg_sfr_csv)

    galaxy_ids = sorted(galaxy_index)
    known_ids = [g for g in galaxy_ids if g in masses]
    n_unknown = len(galaxy_ids) - len(known_ids)
    if n_unknown > 0:
        print(
            'Excluding {0} galaxies with no mass in {1} from the'
            ' plot.'.format(n_unknown, avg_sfr_csv)
        )

    log_mass = {g: np.log10(masses[g]) for g in known_ids}
    log_ssfr = {
        g: np.log10(ssfrs[g]) for g in known_ids if ssfrs[g] > 0.
    }

    groups = {
        'population': known_ids,
        'population excl. test': [
            g for g in known_ids if g not in locked_galaxies
        ],
        'test': [g for g in known_ids if g in locked_galaxies],
    }

    mass_edges = np.histogram_bin_edges(
        list(log_mass.values()), bins=n_bins,
    )
    ssfr_edges = np.histogram_bin_edges(
        list(log_ssfr.values()), bins=n_bins,
    )
    ssfr_bin_width = ssfr_edges[1] - ssfr_edges[0]
    # Placed a few bin widths left of the star-forming range, with a
    # dashed separator, so it reads as a distinct category rather
    # than a point on the continuous log10(ssfr) axis.
    quenched_x = ssfr_edges[0] - 3. * ssfr_bin_width
    separator_x = ssfr_edges[0] - 1.5 * ssfr_bin_width

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Sub-width per group's quenched bar so groups sit side by side
    # inside the quenched slot, rather than fully overlapping (which
    # could hide a shorter bar entirely behind a taller one).
    quenched_bar_width = ssfr_bin_width / len(groups)

    fig, (ax_mass, ax_ssfr) = plt.subplots(1, 2, figsize=(12, 5))
    for i, ((label, ids), color) in enumerate(
            zip(groups.items(), colors)):
        mass_vals = [log_mass[g] for g in ids]
        ax_mass.hist(
            mass_vals,
            bins=mass_edges,
            density=True,
            histtype='step',
            linewidth=2,
            color=color,
            label=r'{0} $(N_{{\mathrm{{gal}}}}={1})$'.format(
                label, len(ids)
            ),
        )

        n_total = len(ids)
        sf_vals = [log_ssfr[g] for g in ids if g in log_ssfr]
        n_quenched = n_total - len(sf_vals)
        # Weight each star-forming galaxy by 1 / (n_total *
        # ssfr_bin_width) rather than passing density=True (which
        # would normalize against only the star-forming count), so
        # this histogram's area is (n_sf / n_total) and lines up with
        # the quenched bar's area of (n_quenched / n_total) below.
        weight = (
            1. / (n_total * ssfr_bin_width) if n_total else 0.
        )
        ax_ssfr.hist(
            sf_vals,
            bins=ssfr_edges,
            weights=[weight] * len(sf_vals),
            histtype='step',
            linewidth=2,
            color=color,
            label=r'{0} ($N_{{\mathrm{{gal}}}}={1}$, {2} quenched)'
                .format(label, n_total, n_quenched),
        )
        quenched_frac = n_quenched / n_total if n_total else 0.
        bar_x = (
            quenched_x
            - ssfr_bin_width / 2.
            + (i + 0.5) * quenched_bar_width
        )
        ax_ssfr.bar(
            bar_x,
            quenched_frac / ssfr_bin_width,
            width=quenched_bar_width,
            color=color,
        )

    legend_anchor = (0.5, -0.15)

    ax_mass.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$')
    ax_mass.set_ylabel('Density')
    ax_mass.set_title('Stellar mass distribution')
    ax_mass.legend(bbox_to_anchor=legend_anchor, loc='upper center')

    ax_ssfr.axvline(separator_x, color='gray', linestyle=':')
    # Explicit ticks replace the default locator, which would
    # otherwise autoscale to the quenched bars' and separator's x
    # positions too and place numeric ticks in that gap, cluttering
    # right where the dashed separator is meant to keep the quenched
    # bars visually distinct from the real log10(ssfr) axis.
    numeric_ticks = [
        t for t in mpl.ticker.MaxNLocator(nbins=6).tick_values(
            ssfr_edges[0], ssfr_edges[-1]
        )
        if ssfr_edges[0] <= t <= ssfr_edges[-1]
    ]
    ax_ssfr.set_xticks([quenched_x] + numeric_ticks)
    ax_ssfr.set_xticklabels(
        ['Quenched'] + ['{0:g}'.format(t) for t in numeric_ticks]
    )
    ax_ssfr.set_xlabel(
        r'$\log_{10}(\mathrm{sSFR} / \mathrm{yr}^{-1})$'
    )
    ax_ssfr.set_ylabel('Density')
    ax_ssfr.set_title('sSFR distribution')
    # bbox_to_anchor's y is negative (below the axes, in axes-fraction
    # coordinates) and loc='upper center' aligns the legend's top edge
    # (not its center) to that anchor, so the legend hangs downward
    # below the plot instead of sitting inside it.
    ax_ssfr.legend(bbox_to_anchor=legend_anchor, loc='upper center')

    fig.tight_layout()
    # tight_layout() doesn't reserve space for a legend placed outside
    # the axes via bbox_to_anchor, so it can get clipped off the
    # bottom of the figure without this.
    fig.subplots_adjust(bottom=0.3)
    plt.show()


def load_gal_for_imshow(gal_id, img_orientation, d):
    """Extract and preprocess one galaxy image for matplotlib imshow.

    Applies min-max scaling to [0, 255], drops the vmap channel, and
    permutes from NCHW to NHWC so imshow interprets the last axis as
    channels.

    Parameters
    ----------
    gal_id : int
        Integer galaxy ID (e.g. 470 for object_470).
    img_orientation : str
        Projection key, e.g. 'projection_xy'.
    d : dict
        Dataset dict returned by preprocessing.load_data. Must contain
        'obs_sorted', 'orientations', and 'X_proc' (the preprocessed
        image tensor of shape (N, C, H, W), populated by the caller
        before invoking any visual_checks function).

    Returns
    -------
    numpy.ndarray of shape (1, H, W, 3), dtype int, or None if the
    requested galaxy and orientation are absent from the dataset.
    """
    import numpy as np
    import torch
    from gallearn import preprocessing

    obj_str = 'object_' + str(gal_id)
    is_obj = d['obs_sorted'] == obj_str
    is_orientation = d['orientations'] == img_orientation
    if np.sum(is_obj & is_orientation) == 0:
        print(f'{obj_str}, {img_orientation} is not in the data')
        return None
    x = d['X_proc'][is_obj & is_orientation]
    x = preprocessing.min_max_scale_255(x)
    # Drop the vmap channel; keep only the 3 RGB channels.
    x = x[:, :3]
    # Dataset is in nchw, c is in rgb (rgu, really). Permute to nhwc so
    # imshow interprets the last axis as channels.
    x = x.permute(0, 2, 3, 1).to(torch.int)
    x = x.cpu().detach().numpy()
    return x


def show_gal_fr_X(gal_id, d):
    """Show all 11 projections of a galaxy in a 3x4 subplot grid.

    Row 0 shows the three standard projections (xy, yz, zx) with x/y/z
    axis labels; the fourth cell in row 0 is hidden. Rows 1 and 2 show
    the eight octant projections with axes hidden and the projection name
    as the subplot title.

    Parameters
    ----------
    gal_id : int
        Integer galaxy ID.
    d : dict
        Dataset dict returned by preprocessing.load_data. Must contain
        'X_proc' (see load_gal_for_imshow).

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    # (projection suffix, xlabel, ylabel) -- empty strings mean hide axes
    standard = [
        ('xy', 'x', 'y'),
        ('yz', 'y', 'z'),
        ('zx', 'z', 'x'),
    ]
    octant_row1 = ['ppp', 'ppm', 'pmp', 'pmm']
    octant_row2 = ['mpp', 'mpm', 'mmp', 'mmm']

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20., 15.))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    for col, (suffix, xlabel, ylabel) in enumerate(standard):
        ax = axs[0, col]
        x = load_gal_for_imshow(gal_id, f'projection_{suffix}', d)
        ax.set_title(suffix)
        if x is not None:
            ax.imshow(x[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
    axs[0, 3].set_visible(False)

    for row_idx, octant_row in enumerate([octant_row1, octant_row2]):
        for col, suffix in enumerate(octant_row):
            ax = axs[row_idx + 1, col]
            x = load_gal_for_imshow(
                gal_id, f'projection_{suffix}', d
            )
            ax.set_title(suffix)
            if x is not None:
                ax.imshow(x[0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


def vmap_vs_image(gal_id, orientation, d):
    """Show the vmap channel alongside the corresponding galaxy image.

    Displays three panels: a pcolormesh of the vmap (seismic_r colormap,
    symmetric vmin/vmax), an imshow of the raw vmap tensor, and an imshow
    of the band_r (channel 0) image in grayscale.

    Parameters
    ----------
    gal_id : int
        Integer galaxy ID.
    orientation : str
        Projection key, e.g. 'projection_yz'.
    d : dict
        Dataset dict returned by preprocessing.load_data. Must contain
        'X_proc' (see load_gal_for_imshow). Channel 3 of X_proc must be
        the vmap channel.

    Returns
    -------
    None
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    obj_str = 'object_' + str(gal_id)
    is_obj = d['obs_sorted'] == obj_str
    is_orientation = d['orientations'] == orientation
    mask = is_obj & is_orientation
    assert mask.sum() == 1
    vmap = d['X_proc'][mask, 3][0]
    img = d['X_proc'][mask, 0][0]
    galname = d['obs_sorted'][mask]
    vmax = np.nanmax(vmap)
    vmin = -1. * vmax

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.pcolormesh(
        # pcolormesh expects the top left to be at [-1, 0].
        torch.flip(vmap, dims=(0,)),
        cmap=plt.cm.seismic_r,
        vmin=vmin,
        vmax=vmax
    )
    fig.suptitle(galname)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')

    axs[1].imshow(vmap)

    ax = axs[2]
    ax.imshow(
        img,
        interpolation='none',
        cmap='gray',
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')

    plt.show()


def vmap_img_overlay(
        gal_id,
        img_orientation,
        gas_min_sden,
        star_min_sden,
        d):
    """Overlay a live-computed vmap on a galaxy image to check alignment.

    Computes a velocity map from raw particle data using
    uci_tools.vel_map.plot, then displays three panels: the raw vmap,
    the galaxy image with the vmap blended on top (opacity proportional
    to distance from white), and the galaxy image alone. Axis labels show
    physical coordinates in kpc.

    Parameters
    ----------
    gal_id : int
        Integer galaxy ID.
    img_orientation : str
        Projection key, e.g. 'projection_yz'. Must be one of
        'projection_xy', 'projection_yz', or 'projection_zx'.
    gas_min_sden : float
        Minimum gas surface density threshold for the vmap computation.
    star_min_sden : float
        Minimum stellar surface density threshold for the vmap
        computation.
    d : dict
        Dataset dict returned by preprocessing.load_data. Must contain
        'X_proc' (see load_gal_for_imshow).

    Returns
    -------
    None
    """
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import uci_tools
    from gallearn import config

    if img_orientation == 'projection_xy':
        horiz_axis, vert_axis = (0, 1)
    elif img_orientation == 'projection_yz':
        horiz_axis, vert_axis = (1, 2)
    elif img_orientation == 'projection_zx':
        horiz_axis, vert_axis = (2, 0)
    else:
        raise ValueError('That `img_orientation` is not an option.')

    obj_str = 'object_' + str(gal_id)
    x = load_gal_for_imshow(gal_id, img_orientation, d)

    super_dir = config.config.get('paths', 'firebox_data_dir')
    firebox_snap = config.config.get('paths', 'firebox_snap')
    obj_path = os.path.join(
        super_dir,
        firebox_snap,
        'particles_within_Rvir_' + obj_str + '.hdf5',
    )

    fov = uci_tools.firebox_io.get_fov(gal_id)

    star_pos, star_vs, star_ms, _ = uci_tools.firebox_io.load_particles(
        'stellar',
        obj_path,
        only_bound=False,
    )
    gas_pos, gas_vs, gas_ms, _ = uci_tools.firebox_io.load_particles(
        'gas',
        obj_path,
        only_bound=False,
    )

    # Apply FOV filter in the requested projection plane.
    star_in_fov = (
        np.linalg.norm(star_pos[:, [horiz_axis, vert_axis]], axis=1)
        <= fov / 2.
    )
    star_pos = star_pos[star_in_fov]
    star_vs = star_vs[star_in_fov]
    star_ms = star_ms[star_in_fov]

    gas_in_fov = (
        np.linalg.norm(gas_pos[:, [horiz_axis, vert_axis]], axis=1)
        <= fov / 2.
    )
    gas_pos = gas_pos[gas_in_fov]
    gas_vs = gas_vs[gas_in_fov]
    gas_ms = gas_ms[gas_in_fov]

    (
        colormesh_gas,
        colormesh_stars,
        x_edges_gas,
        z_edges_gas,
        x_edges_stars,
        z_edges_stars,
        pcol_gas,
        pcol_stars,
    ) = uci_tools.vel_map.plot(
        star_pos,
        star_vs,
        star_ms,
        gas_pos,
        gas_vs,
        gas_ms,
        obj_str,
        '600',
        horiz_axis=horiz_axis,
        vert_axis=vert_axis,
        res=256,
        min_gas_cden=gas_min_sden,
        min_stars_cden=star_min_sden,
        show_plot=False
    )

    norm = plt.Normalize(*pcol_gas.get_clim())

    rgba = torch.tensor(
        pcol_gas.get_cmap()(norm(colormesh_gas)) * 255,
        dtype=torch.int
    )
    rgba_unmod = rgba.clone()

    rgb = rgba[..., :3]
    dist_to_white = torch.linalg.norm(rgb - 255., axis=-1)
    dist_to_white = (
        (dist_to_white - dist_to_white.min())
        / (dist_to_white.max() - dist_to_white.min())
    )
    rgba = rgba.to(torch.float)
    rgba[..., 3] *= dist_to_white ** 0.8
    rgba = rgba.to(torch.int)

    fig = plt.figure(figsize=(20, 10))
    axs = fig.subplots(1, 3)

    axs[0].imshow(
        rgba_unmod,
        interpolation='nearest',
        extent=[
            x_edges_gas.min(),
            x_edges_gas.max(),
            z_edges_gas.min(),
            z_edges_gas.max(),
        ]
    )

    axs[1].imshow(
        x[0],
        extent=[
            x_edges_gas.min(),
            x_edges_gas.max(),
            z_edges_gas.min(),
            z_edges_gas.max(),
        ]
    )
    axs[1].imshow(
        rgba,
        interpolation='nearest',
        alpha=0.6,
        extent=[
            x_edges_gas.min(),
            x_edges_gas.max(),
            z_edges_gas.min(),
            z_edges_gas.max(),
        ]
    )

    axs[2].imshow(
        x[0],
        extent=[
            x_edges_gas.min(),
            x_edges_gas.max(),
            z_edges_gas.min(),
            z_edges_gas.max(),
        ]
    )
    axis_labels = ['x', 'y', 'z']
    for ax in axs:
        ax.set_xlabel(
            '{0} [kpc]'.format(axis_labels[horiz_axis]), fontsize=13
        )
        ax.set_ylabel(
            '{0} [kpc]'.format(axis_labels[vert_axis]), fontsize=13
        )

    plt.show()
