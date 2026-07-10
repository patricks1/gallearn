"""
Training script for galaxy models.

Supports two tasks:
- classifier: binary classification (quenched vs star-forming)
- regressor: sSFR regression (trained on star-forming galaxies only)

And two model architectures:
- bernoulli: BernoulliNet with a torchvision ResNet-18 backbone
- resnet: custom ResNet from cnn.py
"""
import argparse
import datetime
import os

import torch
import torch.nn as nn
import torchvision

from . import cnn
from . import config
from . import preprocessing


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
        'Class balance: {0:.0f} quenched, '
        '{1:.0f} star-forming'.format(n_quenched, n_star_forming)
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
            ' star-forming galaxies'.format(
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
        run_name):
    """
    Create a model based on model_type.

    Parameters
    ----------
    model_type : str
        'bernoulli' or 'resnet'.
    lr : float
        Learning rate.
    dataset : str
        Dataset filename.
    run_name : str
        Name for this training run.

    Returns
    -------
    nn.Module
    """
    # --model and --task are independent axes on purpose. Every model
    # here emits a single scalar, and the task (in main) picks the loss,
    # so any model can pair with any task. Current practice is bernoulli
    # for the classifier and resnet for the regressor, but keeping them
    # decoupled leaves other pairings available to experiment with.
    # cnn.py also defines Net and BottleNeck, not wired in here yet.
    if model_type == 'bernoulli':
        backbone = torchvision.models.resnet18()
        model = cnn.BernoulliNet(
            lr=lr,
            momentum=0.9,
            backbone=backbone,
            dataset=dataset,
            in_channels=4,
        )
    elif model_type == 'resnet':
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
            "model_type must be 'bernoulli' or 'resnet', "
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
        test_loader,
        loss_fn,
        device,
        metrics_fn,
        task):
    """
    Evaluate the model on test data.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    test_loader : DataLoader
        Test data loader.
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

    for images, rs, targets in test_loader:
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
    metrics['loss'] = total_loss / len(test_loader)

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
        test_metrics,
        run_dir,
        train_idxs,
        test_idxs,
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
    test_metrics : dict
        Test metrics for this epoch.
    run_dir : str
        Directory to save checkpoint.
    train_idxs : torch.Tensor
        Training set indices (for reproducible splits).
    test_idxs : torch.Tensor
        Test set indices (for reproducible splits).
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
        'test_metrics': test_metrics,
        'train_idxs': train_idxs.cpu(),
        'test_idxs': test_idxs.cpu(),
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
        task,
        model_type,
        dataset=config.config['gallearn_paths']['dataset'],
        run_name=None,
        n_epochs=100,
        batch_size=32,
        lr=1e-4,
        test_fraction=0.15,
        seed=42,
        wandb_mode='n',
        resume_from=None,
        use_scheduler=True):
    """
    Main training function.

    Parameters
    ----------
    task : str
        'classifier' or 'regressor'. Required.
    model_type : str
        'bernoulli' or 'resnet'. Required.
    dataset : str
        Dataset filename.
    run_name : str, optional
        Name for this run. If None, uses timestamp.
    n_epochs : int
        Number of epochs to train.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    test_fraction : float
        Fraction of data for testing.
    seed : int
        Random seed.
    wandb_mode : str
        'n' for no wandb, 'y' for new run, 'r' for resume.
    resume_from : str, optional
        Path to checkpoint to resume from.
    use_scheduler : bool
        Whether to use a ReduceLROnPlateau scheduler.
    """
    device = get_device()
    print('Using device: {0}'.format(device))

    train_config = {
        'task': task,
        'model_type': model_type,
        'dataset': dataset,
        'batch_size': batch_size,
        'lr': lr,
        'test_fraction': test_fraction,
        'seed': seed,
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

    # Load checkpoint if resuming
    checkpoint = None
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        saved_config = checkpoint.get('train_config', {})
        if saved_config:
            if saved_config.get('dataset') != dataset:
                print(
                    'Warning: dataset mismatch. '
                    'Checkpoint: {0}, '
                    'Current: {1}'.format(
                        saved_config.get('dataset'),
                        dataset,
                    )
                )

    # Initialize wandb if requested (before setting run_name
    # so wandb can generate one).
    if wandb_mode in ('y', 'r'):
        import wandb
        if task == 'classifier':
            project = 'gallearn_quenched_classifier'
        else:
            project = 'sfr_gallearn'
        wandb.init(
            project=project,
            name=run_name,
            config=train_config,
            resume='must' if wandb_mode == 'r' else None,
        )
        if run_name is None:
            run_name = wandb.run.name

    if run_name is None:
        run_name = datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )

    run_dir = os.path.join(
        config.config['gallearn_paths']['project_data_dir'],
        run_name,
    )
    os.makedirs(run_dir, exist_ok=True)
    print('Run directory: {0}'.format(run_dir))

    # Load metadata (not the full image tensor)
    print('Loading metadata...')
    d, N, hdf5_path = preprocessing.load_metadata(dataset)
    print('{0} galaxies in data'.format(N))

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

    # Train/test split over valid_indices only
    N_valid = len(valid_indices)
    N_test = max(1, int(test_fraction * N_valid))
    N_train = N_valid - N_test

    if checkpoint is not None and 'train_idxs' in checkpoint:
        print('Using train/test split from checkpoint')
        train_idxs = checkpoint['train_idxs']
        test_idxs = checkpoint['test_idxs']
    else:
        generator = torch.Generator(device='cpu')
        if seed is not None:
            generator.manual_seed(seed)
        perm = torch.randperm(
            N_valid, generator=generator
        )
        train_idxs = valid_indices[perm[:N_train]]
        test_idxs = valid_indices[perm[N_train:]]

    print(
        'Train: {0}, Test: {1}'.format(
            len(train_idxs), len(test_idxs)
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
    test_dataset = preprocessing.LazyGalaxyDataset(
        hdf5_path,
        test_idxs,
        scaling_means,
        scaling_stds,
        stretch,
        targets[test_idxs],
        rs[test_idxs],
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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=N_workers,
        persistent_workers=N_workers > 0,
        generator=torch.Generator(device='cpu'),
    )

    # Create model
    model = create_model(
        model_type, lr, dataset, run_name
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
                verbose=True,
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
        test_metrics = evaluate(
            model, test_loader, loss_fn, device,
            metrics_fn, task,
        )

        if scheduler is not None:
            scheduler.step(test_metrics['loss'])

        # Logging
        if task == 'classifier':
            # macro_f1 averages F1 across both classes equally, so a
            # model that ignores the minority class (quenched) is
            # penalized even if its star-forming F1 is high.
            print(
                'Epoch {0:3d} | '
                'Train Loss: {1:.4f}, '
                'Acc: {2:.3f} | '
                'Test Loss: {3:.4f}, '
                'Acc: {4:.3f}, '
                'Macro F1: {5:.3f}'.format(
                    epoch,
                    train_metrics['loss'],
                    train_metrics['accuracy'],
                    test_metrics['loss'],
                    test_metrics['accuracy'],
                    test_metrics['macro_f1'],
                )
            )
        else:
            print(
                'Epoch {0:3d} | '
                'Train Loss: {1:.4f}, '
                'RMSE: {2:.4f} | '
                'Test Loss: {3:.4f}, '
                'RMSE: {4:.4f}, '
                'MAE: {5:.4f}'.format(
                    epoch,
                    train_metrics['loss'],
                    train_metrics['rmse'],
                    test_metrics['loss'],
                    test_metrics['rmse'],
                    test_metrics['mae'],
                )
            )

        if wandb_mode in ('y', 'r'):
            import matplotlib.pyplot as plt
            import wandb
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'test/loss': test_metrics['loss'],
                'learning rate': (
                    model.optimizer.param_groups[0]['lr']
                ),
            }
            if task == 'classifier':
                from sklearn.metrics import ConfusionMatrixDisplay
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(
                    test_metrics['labels'],
                    test_metrics['preds'],
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
                    'test/accuracy': (
                        test_metrics['accuracy']
                    ),
                    'test/precision': (
                        test_metrics['precision']
                    ),
                    'test/recall': (
                        test_metrics['recall']
                    ),
                    'test/specificity': (
                        test_metrics['specificity']
                    ),
                    'test/f1': test_metrics['f1'],
                    'test/f1_quenched': (
                        test_metrics['f1_quenched']
                    ),
                    'test/macro_f1': test_metrics['macro_f1'],
                    'test/confusion_matrix': wandb.Image(fig),
                })
                plt.close(fig)
            else:
                log_dict.update({
                    'train/rmse': train_metrics['rmse'],
                    'train/mae': train_metrics['mae'],
                    'test/rmse': test_metrics['rmse'],
                    'test/mae': test_metrics['mae'],
                    'test/r2': test_metrics['r2'],
                })
            wandb.log(log_dict)

        # Save checkpoint if best metric
        test_value = test_metrics[best_metric_key]
        is_best = (
            (higher_is_better and test_value > best_metric)
            or (
                not higher_is_better
                and test_value < best_metric
            )
        )
        if is_best:
            best_metric = test_value
            path = save_checkpoint(
                model,
                epoch,
                train_metrics,
                test_metrics,
                run_dir,
                train_idxs,
                test_idxs,
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
        test_metrics,
        run_dir,
        train_idxs,
        test_idxs,
        train_config,
        target_stats=target_stats,
    )
    print(
        '\nTraining complete. Best test {0}: {1:.4f}'.format(
            best_metric_key, best_metric
        )
    )

    if wandb_mode in ('y', 'r'):
        import wandb
        wandb.finish()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train galaxy models'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['classifier', 'regressor'],
        required=True,
        help=(
            'Task type: classifier (quenched vs'
            ' star-forming) or regressor (sSFR prediction'
            ' on star-forming galaxies)'
        ),
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['bernoulli', 'resnet'],
        required=True,
        help=(
            'Model architecture: bernoulli (torchvision'
            ' ResNet-18 backbone) or resnet (custom'
            ' ResNet)'
        ),
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=config.config['gallearn_paths']['dataset'],
        help=(
            'Dataset filename (default: %(default)s)'
        ),
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help=(
            'Name for this training run'
            ' (default: timestamp)'
        ),
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs (default: %(default)s)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: %(default)s)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: %(default)s)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: %(default)s)',
    )
    parser.add_argument(
        '-w', '--wandb',
        type=str,
        choices=['n', 'y', 'r'],
        default='n',
        help=(
            'Wandb mode: n=none, y=new run, r=resume'
            ' (default: %(default)s)'
        ),
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--no-scheduler',
        action='store_true',
        help='Disable the learning rate scheduler',
    )

    args = parser.parse_args()

    main(
        task=args.task,
        model_type=args.model,
        dataset=args.dataset,
        run_name=args.run_name,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        wandb_mode=args.wandb,
        resume_from=args.resume,
        use_scheduler=not args.no_scheduler,
    )
