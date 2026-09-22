def plot_mass_target_distributions(
        dataset_fname,
        test_lock_path=None,
        avg_sfr_csv=None,
        tgt_type=None,
        n_bins=30):
    """
    Plot log10(Mstar) and target-value histograms comparing the full
    galaxy population, the population excluding the locked test set,
    and the locked test set itself, so a stratified test lock's
    representativeness is visible directly rather than only implied
    by the lock file's recorded bin_allocations counts.

    The dataset says which target it holds, and target_specs says how
    to draw it, so this works for whichever target the file carries
    rather than assuming sSFR. A target whose spec sets a
    plot_scale of 'log' or 'logit' gets two target panels instead of
    one: a linear one showing every galaxy, and a second one
    transformed onto that scale (log10 for 'log', log-odds for
    'logit'), which represents each group's values the transform
    can't show (non-positive for 'log'; outside (0, 1) for 'logit')
    as a single bar at the left, marked to say so, instead of
    placing them on the continuous axis. For sSFR that bar matches
    how stratify_galaxies treats quenched galaxies as their own
    stratum; for gas fraction it separates out the small gas-free
    population the same way. That bar and the scaled panel's
    continuous histogram share one normalization per group (both
    divide by the group's total galaxy count), so their areas sum to
    1 and are directly comparable. A target with plot_scale 'linear'
    gets only the one linear panel, since a linear axis already
    shows its population's shape fine and a second transformed
    panel would add nothing.

    Excludes galaxies missing from avg_sfr_csv from every panel
    (their mass is unknown).

    Parameters
    ----------
    dataset_fname : str
        Training HDF5 filename, as passed to
        preprocessing.load_metadata. Must already be locked via
        gallearn.dataset_lock.lock_dataset (e.g. via
        scripts/lock_dataset.py). Its recorded tgt_type decides how
        the target panel gets drawn.
    test_lock_path : str or pathlib.Path, optional
        Path to a specific test_lock_v<N>.json. Defaults to
        splitting.latest_test_lock_path(), the highest version
        currently in splitting.SPLITS_DIR.
    avg_sfr_csv : str or pathlib.Path, optional
        Path to the avg_sfrs CSV, passed to
        splitting.load_avg_sfr_csv. Defaults to
        splitting.AVG_SFR_CSV. Read for stellar masses only,
        whatever target the dataset holds.
    tgt_type : str, optional
        Which target the dataset holds, for datasets built before
        src/Dataset.jl recorded that in the HDF5 itself. Must be a
        key of target_specs.REGISTRY: 'sfr', 'avg_sfr', 'fgas', or
        'fdm'. Those datasets require it, since nothing else says
        what their values mean. Passing it for a dataset that does
        declare its own target is an error when the two disagree,
        matching how gallearn.train.main guards --target.
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
    from . import target_specs

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

    declared = d['tgt_type']
    if declared is not None and tgt_type is not None:
        if declared != tgt_type:
            raise ValueError(
                'Dataset {0!r} declares tgt_type {1!r}, but this'
                ' call passed {2!r}. Omit tgt_type for a dataset'
                ' that declares its own.'.format(
                    dataset_fname, declared, tgt_type
                )
            )
    if declared is not None:
        tgt_type = declared
    elif tgt_type is None:
        raise ValueError(
            'Dataset {0!r} does not record which target it holds, so'
            ' this plot cannot say what its values mean. It predates'
            ' src/Dataset.jl recording tgt_type, so pass tgt_type'
            ' explicitly.'.format(dataset_fname)
        )
    spec = target_specs.get(tgt_type)

    galaxy_index = splitting.build_galaxy_index(d['obs_sorted'][:N])
    values = splitting.galaxy_target_values(
        galaxy_index, d['ys_sorted'][:N]
    )
    masses = splitting.load_avg_sfr_csv(avg_sfr_csv)

    galaxy_ids = sorted(galaxy_index)
    known_ids = [g for g in galaxy_ids if g in masses]
    n_unknown = len(galaxy_ids) - len(known_ids)
    if n_unknown > 0:
        print(
            'Excluding {0} galaxies with no mass in {1} from the'
            ' plot.'.format(n_unknown, avg_sfr_csv)
        )

    # log10 and logit (log-odds) transforms for plot_scale, plus
    # everything about how each renders that a bare transform
    # function can't say: which raw values it can't show, what to
    # call those values' bar and axis label, and where to put ticks.
    # The logit candidates match scripts/evaluate.py's scatter-plot
    # ticks, so the two views speak the same visual language.
    def _log_ticks(edges):
        ticks = [
            t for t in mpl.ticker.MaxNLocator(nbins=6).tick_values(
                edges[0], edges[-1]
            )
            if edges[0] <= t <= edges[-1]
        ]
        return ticks, ['{0:g}'.format(t) for t in ticks]

    def _logit_ticks(edges):
        candidates = [
            .001, .01, .05, .1, .3, .5, .7, .9, .95, .99, .999,
        ]
        ticks, labels = [], []
        for frac in candidates:
            pos = np.log(frac / (1. - frac))
            if edges[0] <= pos <= edges[-1]:
                ticks.append(pos)
                labels.append('{0:g}'.format(frac))
        return ticks, labels

    def _place_legends_below_xlabels(
            fig, legends, pad_pts=8., gap_frac=0.02, max_iter=4):
        """
        Anchor each (ax, legend) pair's legend just below that axis's
        actual rendered xlabel, then grow the figure's bottom margin
        until the lowest legend clears the figure's edge.

        Measuring the real label instead of guessing a fixed offset
        is what lets this handle a label of any length or rotation
        (a plain one-line title, or the scaled panel's rotated
        tick-crowded one) without a per-case magic number. Iterates
        because changing the bottom margin changes every axes'
        height, which changes what each axes-fraction anchor below
        maps to in figure coordinates, including the one just placed.
        """
        for _ in range(max_iter):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax, legend in legends:
                label_bbox = ax.xaxis.label.get_window_extent(
                    renderer
                )
                pad_px = pad_pts / 72. * fig.dpi
                y_display = label_bbox.y0 - pad_px
                y_axes = ax.transAxes.inverted().transform(
                    (0., y_display)
                )[1]
                legend.set_bbox_to_anchor(
                    (0.5, y_axes), transform=ax.transAxes
                )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            min_fig_y = min(
                legend.get_window_extent(renderer)
                .transformed(fig.transFigure.inverted()).y0
                for _, legend in legends
            )
            overshoot = gap_frac - min_fig_y
            if abs(overshoot) < 0.005:
                break
            new_bottom = min(
                max(fig.subplotpars.bottom + overshoot, 0.05), 0.6
            )
            fig.subplots_adjust(bottom=new_bottom)

    # Each scale can exclude values from one side (log: non-positive)
    # or two (logit: at or below 0, at or above 1); a boundary is one
    # excluded side, with its own bar, tick, and legend count, since
    # log10(0) and logit(0) both go to -inf but logit(1) goes to
    # +inf, so a single '-inf' label would misdescribe a galaxy
    # excluded for sitting at 1.
    scales = {
        'log': {
            'forward': np.log10,
            'axis_fmt': 'log$_{{10}}$({0}{1})',
            'tick_fn': _log_ticks,
            # log10's ticks land at even integer steps, so they never
            # crowd each other and read fine horizontal.
            'tick_rotation': 0,
            'boundaries': [
                {
                    'side': 'left',
                    'excludes': lambda x: x <= 0.,
                    'tick': r'$-\infty$',
                    'legend': 'at zero',
                },
            ],
        },
        'logit': {
            'forward': lambda x: np.log(x / (1. - x)),
            'axis_fmt': 'logit({0}{1})',
            'tick_fn': _logit_ticks,
            # Unlike log10, logit's fixed fraction candidates land at
            # uneven, sometimes tight, log-odds spacing near each end
            # (0.9/0.95/0.99/0.999 all sit close together); rotate so
            # neighboring labels don't overlap.
            'tick_rotation': 45,
            'boundaries': [
                {
                    'side': 'left',
                    'excludes': lambda x: x <= 0.,
                    'tick': r'$-\infty$',
                    'legend': 'at 0',
                },
                {
                    'side': 'right',
                    'excludes': lambda x: x >= 1.,
                    'tick': r'$+\infty$',
                    'legend': 'at 1',
                },
            ],
        },
    }
    for _scale in scales.values():
        _boundaries = _scale['boundaries']
        _scale['valid'] = lambda x, bs=_boundaries: not any(
            b['excludes'](x) for b in bs
        )

    log_mass = {g: np.log10(masses[g]) for g in known_ids}
    # Always available: every galaxy, zeros included. This is the
    # only target data a linear panel needs, whether that panel is
    # the target's sole panel (plot_scale 'linear') or the extra
    # linear panel shown beside a scaled one (plot_scale 'log' or
    # 'logit').
    linear_plotted = {g: values[g] for g in known_ids}
    # A non-linear plot_scale drops the values its transform can't
    # show out of the continuous axis and counts them separately
    # below, via scale_plotted here and the excluded-bar block
    # further down.
    if spec.plot_scale != 'linear':
        scale = scales[spec.plot_scale]
        scale_plotted = {
            g: scale['forward'](values[g]) for g in known_ids
            if scale['valid'](values[g])
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
    linear_edges = np.histogram_bin_edges(
        list(linear_plotted.values()), bins=n_bins,
    )
    if spec.plot_scale != 'linear':
        scale_edges = np.histogram_bin_edges(
            list(scale_plotted.values()), bins=n_bins,
        )
        scale_bin_width = scale_edges[1] - scale_edges[0]
        # Each boundary gets an excluded slot a few bin widths beyond
        # the plotted range, on the side it belongs to, with a dashed
        # separator so it reads as a distinct category rather than a
        # point on the continuous axis.
        boundary_x = {}
        for boundary in scale['boundaries']:
            if boundary['side'] == 'left':
                edge = scale_edges[0]
                sign = -1.
            else:
                edge = scale_edges[-1]
                sign = 1.
            boundary_x[boundary['side']] = {
                'excluded_x': edge + sign * 3. * scale_bin_width,
                'separator_x': edge + sign * 1.5 * scale_bin_width,
            }
        # Sub-width per group's bar so groups sit side by side inside
        # an excluded slot, rather than fully overlapping (which
        # could hide a shorter bar entirely behind a taller one).
        excluded_bar_width = scale_bin_width / len(groups)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # A non-linear plot_scale gets three panels (mass, linear target,
    # scaled target) laid out left to right so the linear view sits
    # right beside the scaled one it complements; otherwise just mass
    # and the target's one linear panel.
    n_panels = 3 if spec.plot_scale != 'linear' else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    ax_mass = axes[0]
    ax_tgt_linear = axes[1]
    ax_tgt_scaled = axes[2] if spec.plot_scale != 'linear' else None

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
        linear_vals = [linear_plotted[g] for g in ids]
        ax_tgt_linear.hist(
            linear_vals,
            bins=linear_edges,
            density=True,
            histtype='step',
            linewidth=2,
            color=color,
            label=r'{0} $(N_{{\mathrm{{gal}}}}={1})$'.format(
                label, n_total
            ),
        )

        if spec.plot_scale != 'linear':
            shown_vals = [
                scale_plotted[g] for g in ids if g in scale_plotted
            ]
            # Weight each shown galaxy by
            # 1 / (n_total * scale_bin_width) rather than passing
            # density=True (which would normalize against only the
            # shown count), so this histogram's area is
            # (n_shown / n_total) and lines up with the excluded
            # bars' combined area of (n_excluded / n_total) below.
            weight = (
                1. / (n_total * scale_bin_width) if n_total else 0.
            )
            # One count and bar per boundary, e.g. logit's separate
            # 'at 0' and 'at 1' rather than one combined count that
            # can't say which side a galaxy was excluded from.
            boundary_counts = [
                (
                    boundary,
                    sum(
                        1 for g in ids
                        if boundary['excludes'](values[g])
                    ),
                )
                for boundary in scale['boundaries']
            ]
            hist_label = (
                r'{0} ($N_{{\mathrm{{gal}}}}={1}$, {2})'.format(
                    label, n_total,
                    ', '.join(
                        '{0} {1}'.format(n_b, boundary['legend'])
                        for boundary, n_b in boundary_counts
                    ),
                )
            )
            ax_tgt_scaled.hist(
                shown_vals,
                bins=scale_edges,
                weights=[weight] * len(shown_vals),
                histtype='step',
                linewidth=2,
                color=color,
                label=hist_label,
            )
            for boundary, n_b in boundary_counts:
                pos = boundary_x[boundary['side']]
                frac_b = n_b / n_total if n_total else 0.
                bar_x = (
                    pos['excluded_x']
                    - scale_bin_width / 2.
                    + (i + 0.5) * excluded_bar_width
                )
                ax_tgt_scaled.bar(
                    bar_x,
                    frac_b / scale_bin_width,
                    width=excluded_bar_width,
                    color=color,
                )

    ax_mass.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$')
    ax_mass.set_ylabel('Density')
    ax_mass.set_title('Distribution of stellar mass')
    # Placed at loc='upper center' for now; _place_legends_below_
    # xlabels moves every legend to its actual rendered xlabel's
    # bottom edge once all three panels' labels are set, rather than
    # guessing a fixed offset that only fits an unrotated one-line
    # label.
    mass_legend = ax_mass.legend(loc='upper center')

    # Built in text mode with math only around the pieces that need
    # it, the way scripts/evaluate.py writes its axis labels. A
    # spec's unit is itself a mixed fragment (sSFR's is 'yr$^{-1}$'),
    # so wrapping the whole label in $...$ would nest delimiters and
    # mathtext would render the label wrong.
    unit_suffix = ' ({0})'.format(spec.unit) if spec.unit else ''
    linear_xlabel = '{0}{1}'.format(spec.axis_label, unit_suffix)
    # The spec spells its own target, so this passes axis_label
    # through untouched. str.capitalize would rewrite 'sSFR' as
    # 'Ssfr', and scripts/evaluate.py already treats the label as
    # authoritative.
    linear_title = 'Distribution of {0}'.format(spec.axis_label)
    if spec.plot_scale != 'linear':
        # Two target panels share one title otherwise, so suffix
        # each to say which axis it's on.
        linear_title += ' (linear)'

    ax_tgt_linear.set_xlabel(linear_xlabel)
    ax_tgt_linear.set_ylabel('Density')
    ax_tgt_linear.set_title(linear_title)
    linear_legend = ax_tgt_linear.legend(loc='upper center')
    legends = [(ax_mass, mass_legend), (ax_tgt_linear, linear_legend)]

    if spec.plot_scale != 'linear':
        for pos in boundary_x.values():
            ax_tgt_scaled.axvline(
                pos['separator_x'], color='gray', linestyle=':'
            )
        # Explicit ticks replace the default locator, which would
        # otherwise autoscale to the excluded bars' and separators' x
        # positions too and place numeric ticks in that gap,
        # cluttering right where the dashed separators are meant to
        # keep those bars visually distinct from the real axis.
        numeric_ticks, numeric_labels = scale['tick_fn'](scale_edges)
        left_boundaries = [
            b for b in scale['boundaries'] if b['side'] == 'left'
        ]
        right_boundaries = [
            b for b in scale['boundaries'] if b['side'] == 'right'
        ]
        all_ticks = (
            [boundary_x[b['side']]['excluded_x']
             for b in left_boundaries]
            + numeric_ticks
            + [boundary_x[b['side']]['excluded_x']
               for b in right_boundaries]
        )
        all_labels = (
            [b['tick'] for b in left_boundaries]
            + numeric_labels
            + [b['tick'] for b in right_boundaries]
        )
        ax_tgt_scaled.set_xticks(all_ticks)
        rotation = scale['tick_rotation']
        ax_tgt_scaled.set_xticklabels(
            all_labels,
            rotation=rotation,
            ha='right' if rotation else 'center',
            # 'anchor' pins the label's corner (rather than its
            # center) to the tick, which is what lets tight_layout
            # below correctly measure how far a rotated label
            # actually reaches and reserve room for it.
            rotation_mode='anchor' if rotation else None,
        )

        scale_unit_suffix = (
            ' / {0}'.format(spec.unit) if spec.unit else ''
        )
        ax_tgt_scaled.set_xlabel(
            scale['axis_fmt'].format(
                spec.axis_label, scale_unit_suffix
            )
        )
        ax_tgt_scaled.set_ylabel('Density')
        ax_tgt_scaled.set_title(
            'Distribution of {0} ({1})'.format(
                spec.axis_label, spec.plot_scale
            )
        )
        scaled_legend = ax_tgt_scaled.legend(loc='upper center')
        legends.append((ax_tgt_scaled, scaled_legend))

    fig.tight_layout()
    # tight_layout() doesn't reserve space for a legend placed outside
    # the axes via bbox_to_anchor, so each legend still needs an
    # explicit position and the figure still needs extra bottom room
    # reserved for it, or it gets clipped off the bottom of the
    # figure.
    _place_legends_below_xlabels(fig, legends)
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
