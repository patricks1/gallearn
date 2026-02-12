"""
Training script for BernoulliNet galaxy classifier.

Classifies galaxies as quenched (0) or star-forming (1) based on their
images and effective radii.
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


def compute_metrics(outputs, labels):
    """
    Compute classification metrics.

    Parameters
    ----------
    outputs : torch.Tensor
        Raw model outputs (logits).
    labels : torch.Tensor
        Ground truth binary labels.

    Returns
    -------
    dict with accuracy, precision, recall, f1
    """
    with torch.no_grad():
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'f1': f1.item(),
    }


def train_epoch(model, train_loader, loss_fn, device):
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

    Returns
    -------
    dict with loss and metrics
    """
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_labels = []

    for images, rs, labels in train_loader:
        images = images.to(device)
        rs = rs.to(device)
        labels = labels.to(device)

        model.optimizer.zero_grad()
        outputs = model(images, rs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        model.optimizer.step()

        total_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(train_loader)

    return metrics


@torch.no_grad()
def evaluate(model, test_loader, loss_fn, device):
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

    Returns
    -------
    dict with loss and metrics
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []

    for images, rs, labels in test_loader:
        images = images.to(device)
        rs = rs.to(device)
        labels = labels.to(device)

        outputs = model(images, rs)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item()
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(test_loader)

    preds = (torch.sigmoid(all_outputs) > 0.5).int()
    metrics['preds'] = preds.cpu().flatten().tolist()
    metrics['labels'] = all_labels.cpu().flatten().int().tolist()

    return metrics


def save_checkpoint(
        model,
        epoch,
        train_metrics,
        test_metrics,
        run_dir,
        train_idxs,
        test_idxs,
        train_config):
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
        Training configuration (dataset, lr, threshold, etc.).

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
        # Save split indices for exact reproducibility on resume
        'train_idxs': train_idxs.cpu(),
        'test_idxs': test_idxs.cpu(),
        # Save config to verify consistency on resume
        'train_config': train_config,
    }
    path = os.path.join(run_dir, f'checkpoint_epoch{epoch:03d}.pt')
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
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return checkpoint


def weights_init(module):
    """Kaiming He initialization for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def main(
        dataset='gallearn_data_256x256_3proj_wsat_wvmap_avg_sfr_tgt.h5',
        run_name=None,
        n_epochs=100,
        batch_size=32,
        lr=1e-4,
        test_fraction=0.15,
        seed=42,
        wandb_mode='n',
        resume_from=None):
    """
    Main training function.

    Parameters
    ----------
    dataset : str
        Dataset filename.
    run_name : str, optional
        Name for this training run. If None, uses timestamp.
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
    """
    # Setup
    device = get_device()
    print(f'Using device: {device}')

    # Training config to save in checkpoint
    train_config = {
        'dataset': dataset,
        'batch_size': batch_size,
        'lr': lr,
        'test_fraction': test_fraction,
        'seed': seed,
    }

    # Load checkpoint if resuming
    checkpoint = None
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        # Use config from checkpoint to ensure consistency
        saved_config = checkpoint.get('train_config', {})
        if saved_config:
            if saved_config.get('dataset') != dataset:
                print(
                    f"Warning: dataset mismatch. "
                    f"Checkpoint: {saved_config.get('dataset')}, "
                    f"Current: {dataset}"
                )

    # Initialize wandb if requested (before setting run_name
    # so wandb can generate one).
    if wandb_mode in ('y', 'r'):
        import wandb
        wandb.init(
            project='gallearn_quenched_classifier',
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
        run_name
    )
    os.makedirs(run_dir, exist_ok=True)
    print('Run directory: {0}'.format(run_dir))

    # Load metadata (not the full image tensor)
    print('Loading metadata...')
    d, N, hdf5_path = preprocessing.load_metadata(dataset)
    print('{0} galaxies in data'.format(N))

    # Binary labels from sSFR
    ssfr = d['ys_sorted'][:N]
    labels = (ssfr > 0).float()

    n_star_forming = labels.sum().item()
    n_quenched = len(labels) - n_star_forming
    print(
        'Class balance: {0:.0f} quenched, '
        '{1:.0f} star-forming'.format(n_quenched, n_star_forming)
    )

    rs = cnn.get_radii(d)[:N]

    # Compute per-channel scaling stats via chunked pass
    # over HDF5 (never loads the full image tensor).
    print('Computing scaling stats...')
    stretch = 1.e-5
    scaling_means, scaling_stds = (
        preprocessing.compute_scaling_stats(hdf5_path, N)
    )

    # Train/test split
    N_test = max(1, int(test_fraction * N))
    N_train = N - N_test

    if checkpoint is not None and 'train_idxs' in checkpoint:
        print('Using train/test split from checkpoint')
        train_idxs = checkpoint['train_idxs']
        test_idxs = checkpoint['test_idxs']
    else:
        generator = torch.Generator(device='cpu')
        if seed is not None:
            generator.manual_seed(seed)
        idxs = torch.randperm(N, generator=generator)
        train_idxs = idxs[:N_train]
        test_idxs = idxs[N_train:]

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
        labels[train_idxs],
        rs[train_idxs],
    )
    test_dataset = preprocessing.LazyGalaxyDataset(
        hdf5_path,
        test_idxs,
        scaling_means,
        scaling_stds,
        stretch,
        labels[test_idxs],
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
    backbone = torchvision.models.resnet18()
    model = cnn.BernoulliNet(
        lr=lr,
        momentum=0.9,
        backbone=backbone,
        dataset=dataset,
        in_channels=4,
    ).to(device)

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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {checkpoint["epoch"]}')
    else:
        model.apply(weights_init)

    print(f'\nModel architecture:\n{model}\n')

    # Loss function (BCEWithLogitsLoss is more numerically stable than
    # Sigmoid + BCELoss)
    loss_fn = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # Training loop
    print('Starting training...\n')
    best_f1 = 0.0

    for epoch in range(start_epoch, start_epoch + n_epochs):
        train_metrics = train_epoch(model, train_loader, loss_fn, device)
        test_metrics = evaluate(model, test_loader, loss_fn, device)

        # Update learning rate
        scheduler.step(test_metrics['loss'])

        # Logging
        print(
            f'Epoch {epoch:3d} | '
            f'Train Loss: {train_metrics["loss"]:.4f}, '
            f'Acc: {train_metrics["accuracy"]:.3f} | '
            f'Test Loss: {test_metrics["loss"]:.4f}, '
            f'Acc: {test_metrics["accuracy"]:.3f}, '
            f'F1: {test_metrics["f1"]:.3f}'
        )

        if wandb_mode in ('y', 'r'):
            import wandb
            cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_metrics['labels'],
                preds=test_metrics['preds'],
                class_names=['quenched', 'star-forming'],
            )
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/f1': train_metrics['f1'],
                'test/loss': test_metrics['loss'],
                'test/accuracy': test_metrics['accuracy'],
                'test/precision': test_metrics['precision'],
                'test/recall': test_metrics['recall'],
                'test/specificity': test_metrics['specificity'],
                'test/f1': test_metrics['f1'],
                'test/confusion_matrix': cm,
                'lr': model.optimizer.param_groups[0]['lr'],
            })

        # Save checkpoint if best F1
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            path = save_checkpoint(
                model,
                epoch,
                train_metrics,
                test_metrics,
                run_dir,
                train_idxs,
                test_idxs,
                train_config,
            )
            print(f'  -> New best F1! Saved checkpoint to {path}')

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
    )
    print(f'\nTraining complete. Best test F1: {best_f1:.3f}')

    if wandb_mode in ('y', 'r'):
        import wandb
        wandb.finish()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train BernoulliNet galaxy classifier'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='gallearn_data_256x256_3proj_wsat_wvmap_avg_sfr_tgt_nchw.h5',
        help='Dataset filename'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this training run'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '-w', '--wandb',
        type=str,
        choices=['n', 'y', 'r'],
        default='n',
        help='Wandb mode: n=none, y=new run, r=resume'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        run_name=args.run_name,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        wandb_mode=args.wandb,
        resume_from=args.resume,
    )
