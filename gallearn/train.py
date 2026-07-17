"""
Training script for galaxy models.

Supports two tasks:
- classifier: binary classification (quenched vs star-forming)
- regressor: sSFR regression (trained on star-forming galaxies only)

And two model architectures:
- standard: StandardNet, a torchvision ResNet-18 backbone
- resnet: custom ResNet from cnn.py
"""
import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

from . import cnn
from . import config
from . import dataset_lock
from . import preprocessing
from . import splitting


def get_device():
    """Select the best available device (MPS, CUDA, or CPU)."""
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except AttributeError:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_classification_metrics(outputs, targets):
    """
    Compute classification metrics.

    Parameters
    ----------
    outputs : torch.Tensor
        Raw model outputs (logits).
    targets : torch.Tensor
        Ground truth binary labels.

    Returns
    -------
    dict with accuracy, precision, recall, f1, specificity
    """
    with torch.no_grad():
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (
            2 * precision * recall / (precision + recall + 1e-8)
        )
        specificity = tn / (tn + fp + 1e-8)

        # F1 for the quenched (negative) class
        precision_q = tn / (tn + fn + 1e-8)
        recall_q = tn / (tn + fp + 1e-8)
        f1_q = (
            2 * precision_q * recall_q
            / (precision_q + recall_q + 1e-8)
        )
        macro_f1 = (f1 + f1_q) / 2

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'f1': f1.item(),
        'f1_quenched': f1_q.item(),
        'macro_f1': macro_f1.item(),
    }


def compute_regression_metrics(outputs, targets):
    """
    Compute regression metrics.

    Parameters
    ----------
    outputs : torch.Tensor
        Model predictions.
    targets : torch.Tensor
        Ground truth values.

    Returns
    -------
    dict with rmse, mae, r2
    """
    with torch.no_grad():
        residuals = outputs - targets
        mse = (residuals ** 2).mean()
        rmse = torch.sqrt(mse)
        mae = residuals.abs().mean()
        ss_res = (residuals ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return {
        'rmse': rmse.item(),
        'mae': mae.item(),
        'r2': r2.item(),
    }


def prepare_targets(task, d, N):
    """
    Prepare targets and valid indices based on task type.

    Parameters
    ----------
    task : str
        'classifier' or 'regressor'.
    d : dict
        Metadata dictionary from load_metadata.
    N : int
        Number of samples in dataset.

    Returns
    -------
    targets : torch.Tensor
        Target values, length N. For the regressor, only entries
        at valid_indices contain meaningful values.
    valid_indices : torch.Tensor
        HDF5 indices to use for training/testing. For classifier
        this is all N indices; for regressor this is the
        star-forming subset.
    target_stats : dict or None
        For regressor: {'means': ..., 'stds': ..., 'stretch': ...}
        needed to invert predictions. None for classifier.
    """
    ssfr = d['ys_sorted'][:N]
    sf_mask = (ssfr > 0).squeeze()

    n_star_forming = sf_mask.sum().item()
    n_quenched = N - n_star_forming
    print(
        'Class balance: {0:.0f} quenched images, '
        '{1:.0f} star-forming images'.format(
            n_quenched, n_star_forming
        )
    )

    if task == 'classifier':
        targets = sf_mask.float().unsqueeze(-1)
        valid_indices = torch.arange(N)
        target_stats = None
    elif task == 'regressor':
        valid_indices = torch.where(sf_mask)[0]
        ssfr_sf = ssfr[valid_indices]
        scaled, means, stds = preprocessing.std_asinh(
            ssfr_sf,
            1.e11,
            return_distrib=True,
        )
        targets = torch.zeros_like(ssfr)
        targets[valid_indices] = scaled
        target_stats = {
            'means': means,
            'stds': stds,
            'stretch': 1.e11,
        }
        print(
            'Regressor: training on {0:.0f}'
            ' star-forming galaxy images'.format(
                len(valid_indices)
            )
        )
    else:
        raise ValueError(
            "task must be 'classifier' or 'regressor', "
            "got '{0}'".format(task)
        )

    return targets, valid_indices, target_stats


def create_model(
        model_type,
        lr,
        dataset,
        run_name,
        pretrained=False):
    """
    Create a model based on model_type.

    Parameters
    ----------
    model_type : str
        'standard' or 'resnet'.
    lr : float
        Learning rate.
    dataset : str
        Dataset filename.
    run_name : str
        Name for this training run.
    pretrained : bool
        Only affects model_type='standard'. If True, its ResNet-18
        backbone starts from ImageNet weights instead of a random
        init, with the pretrained conv1 filters preserved for the
        channels they already cover (see
        cnn._replace_first_conv_in_channels). cnn.ResNet has no
        pretrained option, since it isn't a standard torchvision
        architecture with published weights.

    Returns
    -------
    nn.Module
    """
    # --model and --task are independent axes on purpose. Every model
    # here emits a single scalar, and the task (in main) picks the loss,
    # so any model can pair with any task. Current practice is standard
    # for the classifier and resnet for the regressor, but keeping them
    # decoupled leaves other pairings available to experiment with.
    # cnn.py also defines Net and BottleNeck, not wired in here yet.
    if model_type == 'standard':
        if pretrained:
            backbone = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT,
            )
        else:
            backbone = torchvision.models.resnet18()
        model = cnn.StandardNet(
            lr=lr,
            momentum=0.9,
            backbone=backbone,
            dataset=dataset,
            in_channels=4,
            pretrained=pretrained,
        )
    elif model_type == 'resnet':
        if pretrained:
            raise ValueError(
                "pretrained has no effect on model_type='resnet',"
                " since it isn't a standard architecture with"
                " published weights. Use model_type='standard' for"
                " a pretrained run."
            )
        model = cnn.ResNet(
            run_name,
            N_out_channels=1,
            lr=lr,
            momentum=0.9,
            resblock=cnn.BasicResBlock,
            n_blocks_list=[1, 1, 1, 1],
            dataset=dataset,
            out_channels_list=[16, 32, 64, 128],
            N_img_channels=4,
            auto_load=False,
        )
    else:
        raise ValueError(
            "model_type must be 'standard' or 'resnet', "
            "got '{0}'".format(model_type)
        )
    return model


def train_epoch(
        model,
        train_loader,
        loss_fn,
        device,
        metrics_fn):
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    loss_fn : callable
        Loss function.
    device : torch.device
        Device to use.
    metrics_fn : callable
        Function to compute metrics from (outputs, targets).

    Returns
    -------
    dict with loss and metrics
    """
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for images, rs, targets in train_loader:
        images = images.to(device)
        rs = rs.to(device)
        targets = targets.to(device)

        model.optimizer.zero_grad()
        outputs = model(images, rs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        model.optimizer.step()

        total_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_targets.append(targets.detach())

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = metrics_fn(all_outputs, all_targets)
    metrics['loss'] = total_loss / len(train_loader)

    return metrics


@torch.no_grad()
def evaluate(
        model,
        val_loader,
        loss_fn,
        device,
        metrics_fn,
        task):
    """
    Evaluate the model on validation data.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    val_loader : DataLoader
        Validation data loader.
    loss_fn : callable
        Loss function.
    device : torch.device
        Device to use.
    metrics_fn : callable
        Function to compute metrics from (outputs, targets).
    task : str
        'classifier' or 'regressor'.

    Returns
    -------
    dict with loss and metrics
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for images, rs, targets in val_loader:
        images = images.to(device)
        rs = rs.to(device)
        targets = targets.to(device)

        outputs = model(images, rs)
        loss = loss_fn(outputs, targets)

        total_loss += loss.item()
        all_outputs.append(outputs)
        all_targets.append(targets)

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = metrics_fn(all_outputs, all_targets)
    metrics['loss'] = total_loss / len(val_loader)

    if task == 'classifier':
        preds = (torch.sigmoid(all_outputs) > 0.5).int()
        metrics['preds'] = preds.cpu().flatten().tolist()
        metrics['labels'] = (
            all_targets.cpu().flatten().int().tolist()
        )

    return metrics


def save_checkpoint(
        model,
        epoch,
        train_metrics,
        val_metrics,
        run_dir,
        train_idxs,
        val_idxs,
        train_config,
        target_stats=None):
    """
    Save a training checkpoint.

    Parameters
    ----------
    model : nn.Module
        The model to save.
    epoch : int
        Current epoch number.
    train_metrics : dict
        Training metrics for this epoch.
    val_metrics : dict
        Validation metrics for this epoch.
    run_dir : str
        Directory to save checkpoint.
    train_idxs : torch.Tensor
        Training set indices, resolved from the split file this run
        used (train_config['split_fname']). A resumed run reuses
        these directly rather than re-resolving the split, so
        this checkpoint field is what actually determines a later
        resumed run's training rows, not just a record of them.
    val_idxs : torch.Tensor
        Validation set indices, resolved from the same split file.
        Same as train_idxs: a resumed run reuses this directly.
    train_config : dict
        Training configuration.
    target_stats : dict, optional
        Target scaling stats for regressor (means, stds,
        stretch). None for classifier.

    Returns
    -------
    str : path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_idxs': train_idxs.cpu(),
        'val_idxs': val_idxs.cpu(),
        'train_config': train_config,
    }
    if target_stats is not None:
        checkpoint['target_stats'] = target_stats
    path = os.path.join(
        run_dir, 'checkpoint_epoch{0:03d}.pt'.format(epoch)
    )
    torch.save(checkpoint, path)
    return path


def load_checkpoint(checkpoint_path):
    """
    Load a training checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file.

    Returns
    -------
    dict with checkpoint contents
    """
    checkpoint = torch.load(
        checkpoint_path, weights_only=False
    )
    return checkpoint


def weights_init(module):
    """Kaiming He initialization for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def main(
        task=None,
        model_type=None,
        split_file_path=None,
        run_name=None,
        n_epochs=100,
        batch_size=32,
        lr=1e-3,
        seed=42,
        wandb_mode=None,
        resume_from=None,
        use_scheduler=True,
        pretrained=False,
        train_orientations=None):
    """
    Main training function.

    Parameters
    ----------
    task : str, optional
        'classifier' or 'regressor'. Required for a fresh run; must
        be omitted when resume_from is given, since a resumed run
        reuses the checkpoint's own recorded task.
    model_type : str, optional
        'standard' or 'resnet'. Required for a fresh run; must be
        omitted when resume_from is given, since a resumed run
        reuses the checkpoint's own recorded model_type.
    split_file_path : str, optional
        Path to a train/val split JSON. gallearn.splitting.write_split
        writes this file. A fresh run requires split_file_path
        (main() raises if split_file_path is still None after
        checking resume_from), so every fresh run says explicitly
        which split it uses. Omit split_file_path when passing
        resume_from. The split file's own 'metadata.dataset_path'
        entry determines which dataset this run trains against, so
        there is no separate dataset argument that could disagree
        with it. main() resolves the split's galaxy lists against
        the loaded dataset itself, so the true test set
        (gallearn.splitting.SPLITS_DIR's locked test_lock_v<N>.json
        files) never enters this function. The dataset named in
        split_file_path must already be locked via
        gallearn.dataset_lock.lock_dataset (e.g. via
        scripts/lock_dataset.py); main() raises otherwise.
    run_name : str, optional
        Name for this run. If None on a fresh run, uses a timestamp
        (or wandb's generated name, if wandb_mode='y'). Must be
        omitted when resume_from is given, since a resumed run
        reuses the checkpoint's own recorded run_name (needed to
        find its run directory and, if wandb_mode='r', to resume the
        same wandb run rather than starting a new one).
    n_epochs : int
        Number of epochs to train.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    seed : int
        Random seed.
    wandb_mode : str, optional
        'n' for no wandb, 'y' for a new run. Defaults to 'n' on a
        fresh run. Must be omitted when resume_from is given: a
        resumed run automatically continues its checkpoint's own
        wandb run if it had one (i.e. wandb_run_id is set in its
        train_config), or continues without wandb otherwise. There
        is no way to force wandb on or off for a resumed run
        independent of whether its checkpoint used it, since
        wandb_mode='r' as a distinct caller-chosen mode was a
        footgun: forgetting it silently dropped a stretch of metrics
        with no way to recover them.
    resume_from : str, optional
        Path to checkpoint to resume from. A resumed run always
        reuses the checkpoint's own recorded dataset and
        split_file_path, plus its already-resolved train/val row
        indices, instead of resolving them again, so main() raises
        if the caller also passes split_file_path rather than
        silently ignoring it.
    use_scheduler : bool
        Whether to use a ReduceLROnPlateau scheduler.
    pretrained : bool
        Only affects model_type='standard'. See
        create_model's pretrained parameter.
    train_orientations : list of str, optional
        If given, restricts the training set to rows whose
        orientation is in this list (e.g. ['projection_xy',
        'projection_yz', 'projection_zx']), leaving the val set's
        composition untouched. For ablating how much projection
        diversity affects generalization, holding everything else
        (dataset, split, scaling stats, val set) fixed; not meant
        for routine use, so there is no --train-orientations CLI
        flag, only this argument.
    """
    device = get_device()
    print('Using device: {0}'.format(device))

    # Load checkpoint if resuming
    checkpoint = None
    split_dict = None
    if resume_from is not None:
        if split_file_path is not None:
            raise ValueError(
                'split_file_path has no effect when resume_from is'
                ' given. A resumed run always reuses the'
                ' checkpoint\'s own recorded dataset, split, and'
                ' cached train/val indices. Omit --split when'
                ' passing --resume.'
            )
        if train_orientations is not None:
            raise ValueError(
                'train_orientations has no effect when resume_from'
                ' is given, since a resumed run reuses the'
                ' checkpoint\'s already-filtered cached train_idxs'
                ' rather than rebuilding them.'
            )
        if task is not None or model_type is not None:
            raise ValueError(
                'task and model_type have no effect when resume_from'
                ' is given. A resumed run always reuses the'
                ' checkpoint\'s own recorded task and model_type,'
                ' since building a different architecture than the'
                ' checkpoint\'s weights would either fail to load or'
                ' silently produce a mismatched model. Omit --task'
                ' and --model when passing --resume.'
            )
        if run_name is not None:
            raise ValueError(
                'run_name has no effect when resume_from is given.'
                ' A resumed run always reuses the checkpoint\'s own'
                ' recorded run_name, both to find its run directory'
                ' and, if it had one, to resume the same wandb run.'
                ' Omit --run-name when passing --resume.'
            )
        if wandb_mode is not None:
            raise ValueError(
                'wandb_mode has no effect when resume_from is given.'
                ' A resumed run automatically continues its'
                ' checkpoint\'s own wandb run if it had one, or'
                ' continues without wandb otherwise, so there\'s no'
                ' way to forget to reattach it. Omit --wandb when'
                ' passing --resume.'
            )
        checkpoint = load_checkpoint(resume_from)
        saved_config = checkpoint.get('train_config', {})
        dataset = saved_config.get('dataset')
        split_fname = saved_config.get('split_fname')
        task = saved_config.get('task')
        model_type = saved_config.get('model_type')
        run_name = saved_config.get('run_name')
        wandb_run_id = saved_config.get('wandb_run_id')
        wandb_mode = 'r' if wandb_run_id is not None else 'n'
    else:
        if task is None or model_type is None:
            raise ValueError(
                'task and model_type are required unless resuming'
                ' from a checkpoint that already recorded them.'
            )
        if split_file_path is None:
            raise ValueError(
                'split_file_path is required unless resuming from a'
                ' checkpoint that already recorded one. Create one'
                ' with scripts/split.py split, then pass it with'
                ' --split.'
            )
        wandb_run_id = None
        if wandb_mode is None:
            wandb_mode = 'n'
        with open(split_file_path) as f:
            split_dict = json.load(f)
        dataset = split_dict['metadata']['dataset_path']
        # A resumed run never reopens the split file (see below), so
        # nothing downstream needs the literal path split_file_path
        # was passed in as, which is often absolute and specific to
        # whatever machine started the run. Recording just the
        # filename keeps train_config, checkpoints, and wandb config
        # portable across machines, matching how 'dataset' above is
        # always a bare filename resolved through project_data_dir
        # rather than a stored path.
        split_fname = os.path.basename(split_file_path)

    # Raises if dataset has no recorded hash lock yet, or if its
    # content has drifted since locking, so a checkpoint's cached
    # train_idxs/val_idxs (raw row indices) can't silently end up
    # pointing at different rows than the ones they were resolved
    # against originally.
    dataset_lock.verify_dataset(dataset)

    train_config = {
        'task': task,
        'model_type': model_type,
        'split_fname': split_fname,
        'dataset': dataset,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed,
        'pretrained': pretrained,
        'train_orientations': train_orientations,
        'run_name': run_name,
        'wandb_run_id': wandb_run_id,
    }

    # Task-specific settings
    if task == 'classifier':
        loss_fn = nn.BCEWithLogitsLoss()
        metrics_fn = compute_classification_metrics
        best_metric_key = 'f1'
        higher_is_better = True
    elif task == 'regressor':
        loss_fn = nn.MSELoss()
        metrics_fn = compute_regression_metrics
        best_metric_key = 'loss'
        higher_is_better = False
    else:
        raise ValueError(
            "task must be 'classifier' or 'regressor', "
            "got '{0}'".format(task)
        )

    # Initialize wandb if requested (before setting run_name
    # so wandb can generate one).
    if wandb_mode in ('y', 'r'):
        import wandb
        if task == 'classifier':
            project = 'gallearn_quenched_classifier'
        else:
            project = 'sfr_gallearn'
        if wandb_mode == 'r':
            # wandb resumes by id, not by name; without the
            # original run's id, resume='must' fails even if name
            # matches an existing run, since a name isn't guaranteed
            # unique. wandb_mode is only ever set to 'r' above when
            # the checkpoint recorded a wandb_run_id, so it's always
            # available here.
            wandb.init(
                project=project,
                id=wandb_run_id,
                resume='must',
            )
        else:
            wandb.init(
                project=project,
                name=run_name,
                config=train_config,
            )
            train_config['wandb_run_id'] = wandb.run.id
        if run_name is None:
            run_name = wandb.run.name

    if run_name is None:
        run_name = datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )
    train_config['run_name'] = run_name

    run_dir = os.path.join(
        config.config['gallearn_paths']['project_data_dir'],
        run_name,
    )
    os.makedirs(run_dir, exist_ok=True)
    print('Run directory: {0}'.format(run_dir))

    # Load metadata (not the full image tensor)
    print('Loading metadata...')
    d, N, hdf5_path = preprocessing.load_metadata(dataset)
    print('{0} galaxy images in data'.format(N))

    # Prepare targets based on task
    targets, valid_indices, target_stats = prepare_targets(
        task, d, N
    )
    rs = d['Re'][:N]

    # Compute per-channel image scaling stats via chunked pass
    # over HDF5 (never loads the full image tensor).
    print('Computing scaling stats...')
    stretch = 1.e-5
    scaling_means, scaling_stds = (
        preprocessing.compute_scaling_stats(hdf5_path, N)
    )

    # A resumed run reuses the checkpoint's cached train_idxs/
    # val_idxs rather than re-resolving split_dict, even though
    # dataset_lock.verify_dataset already guarantees the dataset's
    # content: split files, unlike test locks, are not immutable, so
    # a resumed run could otherwise pick up content someone rewrote
    # at split_file_path after the original run (e.g. a rerun of
    # scripts/split.py split with a different seed at the same
    # --output), switching a resumed run's train/val rows
    # mid-experiment with no warning. The checkpoint's cached indices
    # can't drift that way, since a resumed run never loads split_dict
    # at all.
    if checkpoint is not None:
        train_idxs = checkpoint['train_idxs']
        val_idxs = checkpoint['val_idxs']
    else:
        # Resolve the split file's galaxy-level train/val assignment
        # against this dataset's rows, then narrow to valid_indices,
        # so e.g. the regressor still drops a train/val galaxy's
        # quenched rows even though the split itself only knows
        # about galaxies.
        galaxy_index = splitting.build_galaxy_index(
            d['obs_sorted'][:N]
        )
        split_train_idxs, split_val_idxs = (
            splitting.resolve_split_indices(split_dict, galaxy_index)
        )
        valid_mask = torch.zeros(N, dtype=torch.bool)
        valid_mask[valid_indices] = True
        train_idxs = split_train_idxs[valid_mask[split_train_idxs]]
        val_idxs = split_val_idxs[valid_mask[split_val_idxs]]

    if train_orientations is not None:
        orientations = d['orientations'][:N]
        keep_mask = torch.from_numpy(
            np.isin(orientations[train_idxs.numpy()], train_orientations)
        )
        train_idxs = train_idxs[keep_mask]

    print(
        'Train: {0}, Val: {1}'.format(
            len(train_idxs), len(val_idxs)
        )
    )

    # Create lazy datasets that read from HDF5 on demand
    train_dataset = preprocessing.LazyGalaxyDataset(
        hdf5_path,
        train_idxs,
        scaling_means,
        scaling_stds,
        stretch,
        targets[train_idxs],
        rs[train_idxs],
    )
    val_dataset = preprocessing.LazyGalaxyDataset(
        hdf5_path,
        val_idxs,
        scaling_means,
        scaling_stds,
        stretch,
        targets[val_idxs],
        rs[val_idxs],
    )

    N_workers = 4
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=N_workers,
        persistent_workers=N_workers > 0,
        generator=torch.Generator(device='cpu'),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=N_workers,
        persistent_workers=N_workers > 0,
        generator=torch.Generator(device='cpu'),
    )

    # Create model
    model = create_model(
        model_type, lr, dataset, run_name, pretrained=pretrained
    )
    model = model.to(device)

    # Initialize lazy layers with a forward pass
    with torch.no_grad():
        x0, r0, _ = train_dataset[0]
        x1, r1, _ = train_dataset[1]
        sample_X = torch.stack([x0, x1]).to(device)
        sample_rs = torch.stack([r0, r1]).to(device)
        model(sample_X, sample_rs)

    model.init_optimizer()

    # Resume from checkpoint or initialize weights
    start_epoch = 1
    if checkpoint is not None:
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
        model.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        start_epoch = checkpoint['epoch'] + 1
        print(
            'Resumed from epoch {0}'.format(
                checkpoint['epoch']
            )
        )
    else:
        model.apply(weights_init)

    print('\nModel architecture:\n{0}\n'.format(model))

    # For the classifier, recompute loss_fn with pos_weight now that
    # train_idxs is known. pos_weight = n_negative / n_positive makes each
    # positive (star-forming) sample contribute n_neg/n_pos times as much
    # to the loss as a negative (quenched) sample, so the model can't
    # minimize loss by always predicting star-forming.
    if task == 'classifier':
        train_targets = targets[train_idxs]
        n_pos = float((train_targets == 1).sum())
        n_neg = float((train_targets == 0).sum())
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(n_neg / n_pos, device=device)
        )

    # Learning rate scheduler
    if use_scheduler:
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                model.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
        )
    else:
        scheduler = None

    # Training loop
    print('Starting training...\n')
    # higher_is_better is set per-task above: True for classifier (F1),
    # False for regressor (loss). It controls which direction counts as
    # an improvement when deciding whether to save a checkpoint.
    if higher_is_better:
        best_metric = 0.0
    else:
        best_metric = float('inf')

    for epoch in range(
            start_epoch, start_epoch + n_epochs):
        train_metrics = train_epoch(
            model, train_loader, loss_fn, device,
            metrics_fn,
        )
        val_metrics = evaluate(
            model, val_loader, loss_fn, device,
            metrics_fn, task,
        )

        if scheduler is not None:
            lr_before = scheduler.get_last_lr()[0]
            scheduler.step(val_metrics['loss'])
            lr_after = scheduler.get_last_lr()[0]
            if lr_after != lr_before:
                print(
                    'Epoch {0:3d} | Reducing learning rate to'
                    ' {1:.3e}.'.format(epoch, lr_after)
                )

        # Logging
        if task == 'classifier':
            # macro_f1 averages F1 across both classes equally, so a
            # model that ignores the minority class (quenched) is
            # penalized even if its star-forming F1 is high.
            print(
                'Epoch {0:3d} | '
                'Train Loss: {1:.4f}, '
                'Acc: {2:.3f} | '
                'Val Loss: {3:.4f}, '
                'Acc: {4:.3f}, '
                'Macro F1: {5:.3f}'.format(
                    epoch,
                    train_metrics['loss'],
                    train_metrics['accuracy'],
                    val_metrics['loss'],
                    val_metrics['accuracy'],
                    val_metrics['macro_f1'],
                )
            )
        else:
            print(
                'Epoch {0:3d} | '
                'Train Loss: {1:.4f}, '
                'RMSE: {2:.4f} | '
                'Val Loss: {3:.4f}, '
                'RMSE: {4:.4f}, '
                'MAE: {5:.4f}'.format(
                    epoch,
                    train_metrics['loss'],
                    train_metrics['rmse'],
                    val_metrics['loss'],
                    val_metrics['rmse'],
                    val_metrics['mae'],
                )
            )

        if wandb_mode in ('y', 'r'):
            import matplotlib.pyplot as plt
            import wandb
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'val/loss': val_metrics['loss'],
                'learning rate': (
                    model.optimizer.param_groups[0]['lr']
                ),
            }
            if task == 'classifier':
                from sklearn.metrics import ConfusionMatrixDisplay
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(
                    val_metrics['labels'],
                    val_metrics['preds'],
                    display_labels=['quenched', 'star-forming'],
                    normalize='true',
                    ax=ax,
                )
                ax.set_title('Epoch {0}'.format(epoch))
                log_dict.update({
                    'train/accuracy': (
                        train_metrics['accuracy']
                    ),
                    'train/f1': train_metrics['f1'],
                    'val/accuracy': (
                        val_metrics['accuracy']
                    ),
                    'val/precision': (
                        val_metrics['precision']
                    ),
                    'val/recall': (
                        val_metrics['recall']
                    ),
                    'val/specificity': (
                        val_metrics['specificity']
                    ),
                    'val/f1': val_metrics['f1'],
                    'val/f1_quenched': (
                        val_metrics['f1_quenched']
                    ),
                    'val/macro_f1': val_metrics['macro_f1'],
                    'val/confusion_matrix': wandb.Image(fig),
                })
                plt.close(fig)
            else:
                log_dict.update({
                    'train/rmse': train_metrics['rmse'],
                    'train/mae': train_metrics['mae'],
                    'val/rmse': val_metrics['rmse'],
                    'val/mae': val_metrics['mae'],
                    'val/r2': val_metrics['r2'],
                })
            wandb.log(log_dict)

        # Save checkpoint if best metric
        val_value = val_metrics[best_metric_key]
        is_best = (
            (higher_is_better and val_value > best_metric)
            or (
                not higher_is_better
                and val_value < best_metric
            )
        )
        if is_best:
            best_metric = val_value
            path = save_checkpoint(
                model,
                epoch,
                train_metrics,
                val_metrics,
                run_dir,
                train_idxs,
                val_idxs,
                train_config,
                target_stats=target_stats,
            )
            print(
                '  -> New best {0}! '
                'Saved checkpoint to {1}'.format(
                    best_metric_key, path
                )
            )

    # Save final checkpoint
    save_checkpoint(
        model,
        epoch,
        train_metrics,
        val_metrics,
        run_dir,
        train_idxs,
        val_idxs,
        train_config,
        target_stats=target_stats,
    )
    print(
        '\nTraining complete. Best validation {0}: {1:.4f}'.format(
            best_metric_key, best_metric
        )
    )

    if wandb_mode in ('y', 'r'):
        import wandb
        wandb.finish()

    return model
