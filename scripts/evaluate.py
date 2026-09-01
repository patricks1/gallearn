"""
Generate a PDF report visualizing a trained checkpoint's performance
on a split's val set.

Slides
------
1. Classifier: confusion matrix. Regressor: ground-truth-vs-
   prediction scatter, color-coded by stellar mass.
2+. Sample val images (RGB composites) with true/predicted labels.

Read-only: no optimizer, no training loop, no wandb.
"""
import argparse
import json
import os
import re

import h5py
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import gallearn


def infer_head_widths(model_state_dict):
    """
    Infer model.head's per-layer widths directly from a checkpoint's
    own weights, rather than trusting create_model's hardcoded
    defaults.

    Some experiments (e.g. the heteroscedastic regression head in
    docs/status.md) built their model by monkeypatching create_model
    to shrink the head width or change N_out_channels; train_config
    never recorded those ad hoc changes, so create_model alone can't
    reconstruct them.

    Parameters
    ----------
    model_state_dict : dict

    Returns
    -------
    widths : list of int
        Output width of each Linear layer in model.head, in order.
        The last entry is N_out_channels.
    """
    indices = sorted(
        int(m.group(1))
        for k, v in model_state_dict.items()
        for m in [re.match(r'^head\.(\d+)\.weight$', k)]
        if m is not None and v.dim() == 2
    )
    return [
        model_state_dict['head.{0}.weight'.format(i)].shape[0]
        for i in indices
    ]


def build_head(widths):
    """
    Build a head matching cnn.ResNet/StandardNet's own pattern
    (LazyLinear -> BatchNorm1d -> ReLU, repeated, ending in a plain
    Linear), sized to the given per-layer widths.
    """
    layers = [
        nn.LazyLinear(widths[0]),
        nn.BatchNorm1d(widths[0]),
        nn.ReLU(),
    ]
    for prev_width, width in zip(widths[:-1], widths[1:-1]):
        layers += [
            nn.Linear(prev_width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        ]
    layers.append(nn.Linear(widths[-2], widths[-1]))
    return nn.Sequential(*layers)


def load_val_dataset(checkpoint, split_file_path):
    """
    Rebuild the checkpoint's model and the split's val dataset.

    Parameters
    ----------
    checkpoint : dict
        As returned by gallearn.train.load_checkpoint.
    split_file_path : str
        Path to a train/val split JSON.

    Returns
    -------
    model : nn.Module
        Reconstructed model, weights loaded, in eval mode.
    val_dataset : gallearn.preprocessing.LazyGalaxyDataset
    task : str
        'classifier' or 'regressor'.
    galaxy_ids : np.ndarray of str
        d['obs_sorted'] entries for val_idxs, in the same row order
        as val_dataset (i.e. run_inference's outputs/targets).
    ssfr : torch.Tensor
        d['ys_sorted'] entries for val_idxs (raw, unscaled target
        values), same row order as galaxy_ids.
    tgt_type : str
        Which target this checkpoint predicts. See
        gallearn.target_specs.REGISTRY.
    """
    train_config = checkpoint['train_config']
    task = train_config['task']
    model_type = train_config['model_type']
    dataset = train_config['dataset']
    pretrained = train_config.get('pretrained', False)
    run_name = train_config['run_name']

    with open(split_file_path) as f:
        split_dict = json.load(f)

    checkpoint_split_fname = train_config.get('split_fname')
    this_split_fname = os.path.basename(split_file_path)
    if checkpoint_split_fname != this_split_fname:
        print(
            'Warning: evaluating checkpoint trained against split'
            ' {0!r} using a different split {1!r}. Val sets may'
            ' not match.'.format(
                checkpoint_split_fname,
                this_split_fname,
            )
        )

    gallearn.dataset_lock.verify_dataset(dataset)

    print('Loading metadata...')
    d, N, hdf5_path = gallearn.preprocessing.load_metadata(dataset)

    # The checkpoint records the target it trained on. Fall back to
    # the dataset's own declaration for checkpoints written before
    # that was recorded.
    tgt_type = train_config.get('tgt_type') or d['tgt_type']
    if tgt_type is None:
        raise ValueError(
            'Neither this checkpoint nor dataset {0!r} records which'
            ' target it holds, so predictions cannot be mapped back'
            ' to raw units. Both predate that being'
            ' recorded.'.format(dataset)
        )

    valid_indices = gallearn.train.compute_valid_indices(
        task, d, N, tgt_type
    )
    galaxy_index = gallearn.splitting.build_galaxy_index(d['obs_sorted'][:N])
    _, split_val_idxs = gallearn.splitting.resolve_split_indices(
        split_dict,
        galaxy_index,
    )
    valid_mask = torch.zeros(N, dtype=torch.bool)
    valid_mask[valid_indices] = True
    val_idxs = split_val_idxs[valid_mask[split_val_idxs]]

    targets, _, _ = gallearn.train.prepare_targets(
        task,
        d,
        N,
        checkpoint['train_idxs'],
        tgt_type,
        target_stats=checkpoint.get('target_stats'),
    )
    rs = d['Re'][:N]

    print('Validating on {0} images'.format(len(val_idxs)))

    galaxy_ids = d['obs_sorted'][:N][val_idxs.numpy()]
    ssfr = d['ys_sorted'][:N][val_idxs]

    val_dataset = gallearn.preprocessing.LazyGalaxyDataset(
        hdf5_path,
        val_idxs,
        checkpoint['scaling_means'],
        checkpoint['scaling_stds'],
        1.e-5,
        targets[val_idxs],
        rs[val_idxs],
    )

    device = gallearn.train.get_device()
    model = gallearn.train.create_model(
        model_type,
        train_config['lr'],
        dataset,
        run_name,
        pretrained=pretrained,
    )
    head_widths = infer_head_widths(checkpoint['model_state_dict'])
    model.head = build_head(head_widths)
    model = model.to(device)
    with torch.no_grad():
        x0, r0, _ = val_dataset[0]
        x1, r1, _ = val_dataset[1]
        sample_X = torch.stack([x0, x1]).to(device)
        sample_rs = torch.stack([r0, r1]).to(device)
        model(sample_X, sample_rs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, val_dataset, task, galaxy_ids, ssfr, tgt_type


@torch.no_grad()
def run_inference(model, val_dataset, device):
    """Run inference over val_dataset, returning (outputs, targets)."""
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
    )
    all_outputs = []
    all_targets = []
    for images, rs, targets in loader:
        images = images.to(device)
        rs = rs.to(device)
        outputs = model(images, rs)
        all_outputs.append(outputs.cpu())
        all_targets.append(targets)
    return torch.cat(all_outputs), torch.cat(all_targets)


def plot_confusion_matrix(outputs, targets, run_name, pdf):
    """Add a 2x2 confusion matrix page (quenched vs. star-forming)."""
    metrics = gallearn.train.compute_classification_metrics(outputs, targets)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).int().flatten()
    labels = targets.int().flatten()

    cm = np.zeros((2, 2), dtype=int)
    for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
        cm[true_label, pred_label] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    class_names = ['Quenched', 'Star-forming']
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                '{0}\n({1:.1%})'.format(cm[i, j], cm_pct[i, j]),
                ha='center',
                va='center',
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(
        '{0}\nN={1}, F1={2:.3f}, accuracy={3:.3f}'.format(
            run_name,
            len(labels),
            metrics['f1'],
            metrics['accuracy'],
        )
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    print('Metrics: {0}'.format(metrics))


def invert_target_scaling(scaled, target_stats, tgt_type):
    """
    Map scaled targets/predictions back to raw target units.

    The target's spec owns both directions of its scaling, so this
    defers to it rather than reimplementing an inverse here. That
    keeps target_stats opaque: how a given target scales, and what
    it needs to remember to undo that, stays in target_specs.py.

    Parameters
    ----------
    scaled : torch.Tensor
    target_stats : dict
        The spec's scaling statistics, as saved in a checkpoint's
        'target_stats' (see gallearn.train.prepare_targets). Treat
        the contents as opaque.
    tgt_type : str
        Which target these values belong to: 'sfr', 'avg_sfr', or
        'fgas'. See gallearn.target_specs.REGISTRY.

    Returns
    -------
    torch.Tensor, same shape as scaled, in raw target units.
    """
    spec = gallearn.target_specs.get(tgt_type)
    return spec.unscale(scaled, target_stats)


def plot_regression_scatter(
        outputs,
        targets,
        target_stats,
        masses,
        run_name,
        tgt_type,
        pdf):
    """Add a ground-truth-vs-prediction scatter plot page, in raw
    target units (metrics are still computed in scaled space, to
    match the numbers train.py itself logs), color-coded by stellar
    mass.

    Parameters
    ----------
    masses : np.ndarray
        Stellar mass (Msun) per row, aligned with outputs/targets.
        NaN for galaxies missing from the avg_sfrs mass CSV.
    tgt_type : str
        Which target these values belong to. See
        gallearn.target_specs.REGISTRY.
    """
    spec = gallearn.target_specs.get(tgt_type)
    metrics = gallearn.train.compute_regression_metrics(outputs, targets)
    y_true = invert_target_scaling(
        targets, target_stats, tgt_type
    ).flatten().numpy()
    y_pred = invert_target_scaling(
        outputs, target_stats, tgt_type
    ).flatten().numpy()

    # On a log-scaled target, a bad-enough prediction can invert to a
    # non-positive value, which a log axis can't show. Drop those
    # points rather than error or silently clip, and say how many
    # were dropped. A target plotted on linear axes keeps every
    # point, including any legitimate zeros.
    if spec.log_scale:
        positive = (y_true > 0) & (y_pred > 0)
        n_dropped = len(y_true) - positive.sum()
        if n_dropped > 0:
            print(
                'Note: dropping {0} points with non-positive'
                ' predicted {1} (can\'t show on a log-scale'
                ' plot)'.format(n_dropped, spec.axis_label)
            )
        y_true = y_true[positive]
        y_pred = y_pred[positive]
        masses = masses[positive]

    # A galaxy absent from the mass CSV (see
    # gallearn.splitting.load_avg_sfr_csv) has a NaN mass; those
    # points still get plotted (point accuracy doesn't depend on
    # knowing the mass), just without a color, and are counted so
    # missing coverage isn't silently invisible.
    has_mass = ~np.isnan(masses)
    n_no_mass = len(masses) - has_mass.sum()
    if n_no_mass > 0:
        print(
            'Note: {0} points have no stellar mass in the avg_sfrs'
            ' CSV; plotting them uncolored'.format(n_no_mass)
        )

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(
        y_true[~has_mass],
        y_pred[~has_mass],
        s=8,
        alpha=0.4,
        color='lightgray',
        label='no mass' if n_no_mass > 0 else None,
    )
    sc = ax.scatter(
        y_true[has_mass],
        y_pred[has_mass],
        s=8,
        alpha=0.6,
        c=np.log10(masses[has_mass]),
        cmap='viridis',
    )
    fig.colorbar(sc, ax=ax, label=r'log$_{10}$($M_\star$ / M$_\odot$)')
    if n_no_mass > 0:
        ax.legend(loc='upper left', fontsize='small')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    if spec.log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    unit_suffix = ' ({0})'.format(spec.unit) if spec.unit else ''
    ax.set_xlabel(
        'Ground truth {0}{1}'.format(spec.axis_label, unit_suffix)
    )
    ax.set_ylabel(
        'Predicted {0}{1}'.format(spec.axis_label, unit_suffix)
    )
    ax.set_title(
        '{0}\nN={1}, R2={2:.3f}, RMSE={3:.3f}, MAE={4:.3f}'
        ' (scaled space)'.format(
            run_name,
            len(y_true),
            metrics['r2'],
            metrics['rmse'],
            metrics['mae'],
        )
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    print('Metrics (scaled space): {0}'.format(metrics))


def add_sample_slides(
        pdf,
        hdf5_path,
        val_idxs,
        ssfr,
        task,
        outputs,
        target_stats,
        n_samples,
        tgt_type,
        seed=42):
    """
    Add pages of sample val images (RGB composites) with true and
    predicted labels, 5 images per page.

    Classifier: samples split evenly between predicted-quenched and
    predicted-star-forming, so both classes show up regardless of
    class imbalance. Regressor: a uniform random sample.

    Parameters
    ----------
    val_idxs : torch.Tensor
        Row indices into the HDF5, aligned with ssfr/outputs.
    ssfr : torch.Tensor
        Raw (unscaled) sSFR, aligned with val_idxs.
    outputs : torch.Tensor
        Model outputs, already reduced to a single point-prediction
        column for the regressor (heteroscedastic mean/log-variance
        columns handled by the caller, not here).
    target_stats : dict or None
        Needed to invert a regressor's scaled prediction back to raw
        target units; unused for the classifier.
    tgt_type : str
        Which target this run predicts. See
        gallearn.target_specs.REGISTRY.
    """
    spec = gallearn.target_specs.get(tgt_type)
    rng = np.random.default_rng(seed)

    if task == 'classifier':
        probs = torch.sigmoid(outputs).flatten().numpy()
        preds = (probs > 0.5).astype(int)
        n_each = n_samples // 2
        quenched = np.where(preds == 0)[0]
        star_forming = np.where(preds == 1)[0]
        sample_idxs = np.concatenate([
            rng.choice(
                quenched,
                size=min(n_each, len(quenched)),
                replace=False,
            ),
            rng.choice(
                star_forming,
                size=min(n_samples - n_each, len(star_forming)),
                replace=False,
            ),
        ])
    else:
        sample_idxs = rng.choice(
            len(val_idxs),
            size=min(n_samples, len(val_idxs)),
            replace=False,
        )
    rng.shuffle(sample_idxs)

    with h5py.File(hdf5_path, 'r', locking=False) as f:
        for slide_start in range(0, len(sample_idxs), 5):
            slide_idxs = sample_idxs[slide_start:slide_start + 5]
            fig, axes = plt.subplots(
                1, len(slide_idxs), figsize=(14, 3.5),
            )
            if len(slide_idxs) == 1:
                axes = [axes]

            for ax, si in zip(axes, slide_idxs):
                hdf5_idx = val_idxs[si].item()
                # X shape: (N, C, H, W); C order: u, g, r, vmap
                img = f['X'][hdf5_idx]
                r_band = img[2]
                g_band = img[1]
                u_band = img[0]

                # Compose RGB from r, g, u bands using asinh
                # stretch for visualization.
                rgb = np.stack([r_band, g_band, u_band], axis=-1)
                rgb = np.arcsinh(rgb * 1e-5)
                for c in range(3):
                    ch = rgb[:, :, c]
                    lo, hi = np.percentile(ch, [1, 99])
                    if hi > lo:
                        rgb[:, :, c] = np.clip(
                            (ch - lo) / (hi - lo), 0, 1,
                        )
                    else:
                        rgb[:, :, c] = 0.

                ax.imshow(rgb, origin='lower')
                ax.axis('off')

                galaxy_true = ssfr[si].item()
                if task == 'classifier':
                    pred_label = (
                        'star-forming' if preds[si] == 1
                        else 'quenched'
                    )
                    title = (
                        '{0}: {1:.2e} {2}\n'
                        'pred: {3} ({4:.2f})'.format(
                            spec.axis_label, galaxy_true,
                            spec.unit, pred_label, probs[si],
                        )
                    )
                else:
                    pred_val = invert_target_scaling(
                        outputs[si:si + 1], target_stats, tgt_type,
                    ).item()
                    title = (
                        'true: {0:.2e}\n'
                        'pred: {1:.2e} {2}'.format(
                            galaxy_true, pred_val, spec.unit,
                        )
                    )
                ax.set_title(title, fontsize=8)

            fig.suptitle('Sample Val Images', fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main(model_path, split_file_path, output_path=None, n_samples=10):
    checkpoint = gallearn.train.load_checkpoint(model_path)
    run_name = checkpoint['train_config']['run_name']

    (
        model, val_dataset, task, galaxy_ids, ssfr, tgt_type
    ) = load_val_dataset(
        checkpoint,
        split_file_path,
    )
    device = gallearn.train.get_device()
    outputs, targets = run_inference(model, val_dataset, device)

    if output_path is None:
        run_dir = os.path.join(
            gallearn.config.config['gallearn_paths']['project_data_dir'],
            run_name,
        )
        os.makedirs(run_dir, exist_ok=True)
        output_path = os.path.join(run_dir, 'eval_{0}.pdf'.format(task))

    if task == 'regressor' and outputs.shape[1] > 1:
        # A heteroscedastic head (see docs/status.md) outputs
        # (mean, log-variance) columns; only the mean is a point
        # prediction, so that's what the scatter plot and sample
        # slides compare against ground truth. The variance column
        # is dropped here rather than plotted, since calibration is
        # a different question than point accuracy.
        print(
            'Note: model.head outputs {0} columns; using column 0'
            ' as the point prediction (heteroscedastic head --'
            ' ignoring the variance column)'.format(outputs.shape[1])
        )
        outputs = outputs[:, 0:1]

    with matplotlib.backends.backend_pdf.PdfPages(output_path) as pdf:
        if task == 'classifier':
            plot_confusion_matrix(outputs, targets, run_name, pdf)
        elif task == 'regressor':
            masses_by_galaxy = gallearn.splitting.load_avg_sfr_csv()
            masses = np.array([
                masses_by_galaxy.get(gid, np.nan) for gid in galaxy_ids
            ])
            plot_regression_scatter(
                outputs,
                targets,
                checkpoint['target_stats'],
                masses,
                run_name,
                tgt_type,
                pdf,
            )
        else:
            raise ValueError('Unknown task {0!r}'.format(task))

        add_sample_slides(
            pdf,
            val_dataset.hdf5_path,
            val_dataset.indices,
            ssfr,
            task,
            outputs,
            checkpoint.get('target_stats'),
            n_samples,
            tgt_type,
        )

    print('Saved report to {0}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Visualize a checkpoint\'s performance on a split\'s'
            ' val set'
        )
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to a checkpoint written by save_checkpoint',
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        help='Path to a train/val split JSON, from scripts/split.py',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=(
            'Path to save the report PDF. Defaults to'
            ' <project_data_dir>/<run_name>/eval_<task>.pdf'
        ),
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of sample val images to show. Default: 10.',
    )
    args = parser.parse_args()

    main(args.model, args.split, args.output, args.n_samples)
