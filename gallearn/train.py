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
import numpy as np

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


def load_data(dataset_fname, device):
    """
    Load raw data for training (no scaling applied).

    Parameters
    ----------
    dataset_fname : str
        Filename of the HDF5 dataset.
    device : torch.device
        Device to load tensors onto.

    Returns
    -------
    dict with keys: X, rs, labels, and metadata
    """
    d = preprocessing.load_data(dataset_fname)

    X = d['X'].to(device)
    # Note: scaling is applied later using model.scaling_function

    rs = cnn.get_radii(d).to(device)

    # Create binary labels from sSFR
    # ys_sorted contains the target values (sSFR)
    # Quenched (0) = sSFR == 0, Star-forming (1) = sSFR > 0
    ssfr = d['ys_sorted'].to(device)
    labels = (ssfr > 0).float()

    return {
        'X': X,
        'rs': rs,
        'labels': labels,
        'obs_sorted': d['obs_sorted'],
        'orientations': d['orientations'],
    }


def train_test_split(X, rs, labels, test_fraction=0.15, seed=None):
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : torch.Tensor
        Image data, shape (N, C, H, W).
    rs : torch.Tensor
        Effective radii, shape (N, 1).
    labels : torch.Tensor
        Binary labels, shape (N, 1).
    test_fraction : float
        Fraction of data to use for testing.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with train and test splits, plus the indices used
    """
    N = len(labels)
    N_test = max(1, int(test_fraction * N))
    N_train = N - N_test

    generator = torch.Generator(device='cpu')
    if seed is not None:
        generator.manual_seed(seed)

    idxs = torch.randperm(N, generator=generator)
    train_idxs = idxs[:N_train]
    test_idxs = idxs[N_train:]

    # Move indices to same device as data for indexing
    train_idxs_dev = train_idxs.to(X.device)
    test_idxs_dev = test_idxs.to(X.device)

    return {
        'X_train': X[train_idxs_dev],
        'X_test': X[test_idxs_dev],
        'rs_train': rs[train_idxs_dev],
        'rs_test': rs[test_idxs_dev],
        'labels_train': labels[train_idxs_dev],
        'labels_test': labels[test_idxs_dev],
        # Store indices on CPU for checkpoint serialization
        'train_idxs': train_idxs,
        'test_idxs': test_idxs,
    }


def apply_split_indices(X, rs, labels, train_idxs, test_idxs):
    """
    Apply previously saved split indices to data.

    Parameters
    ----------
    X : torch.Tensor
        Image data.
    rs : torch.Tensor
        Effective radii.
    labels : torch.Tensor
        Binary labels.
    train_idxs : torch.Tensor
        Indices for training set.
    test_idxs : torch.Tensor
        Indices for test set.

    Returns
    -------
    dict with train and test splits
    """
    train_idxs_dev = train_idxs.to(X.device)
    test_idxs_dev = test_idxs.to(X.device)

    return {
        'X_train': X[train_idxs_dev],
        'X_test': X[test_idxs_dev],
        'rs_train': rs[train_idxs_dev],
        'rs_test': rs[test_idxs_dev],
        'labels_train': labels[train_idxs_dev],
        'labels_test': labels[test_idxs_dev],
        'train_idxs': train_idxs,
        'test_idxs': test_idxs,
    }


def create_dataloaders(splits, batch_size):
    """
    Create DataLoaders for training and testing.

    Parameters
    ----------
    splits : dict
        Output from train_test_split.
    batch_size : int
        Batch size for training.

    Returns
    -------
    tuple of (train_loader, test_loader)
    """
    train_dataset = torch.utils.data.TensorDataset(
        splits['X_train'],
        splits['rs_train'],
        splits['labels_train']
    )
    test_dataset = torch.utils.data.TensorDataset(
        splits['X_test'],
        splits['rs_test'],
        splits['labels_test']
    )

    # Generators should be on CPU (works with all backends)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device='cpu')
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


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

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
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
):
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
    resume_from=None,
):
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

    if run_name is None:
        run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    run_dir = os.path.join(
        config.config['gallearn_paths']['project_data_dir'],
        run_name
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f'Run directory: {run_dir}')

    # Initialize wandb if requested
    if wandb_mode in ('y', 'r'):
        import wandb
        wandb.init(
            project='gallearn_quenched_classifier',
            name=run_name,
            config=train_config,
            resume='must' if wandb_mode == 'r' else None,
        )

    # Load data (unscaled)
    print('Loading data...')
    data = load_data(dataset, device)
    print(f'Loaded {len(data["X"])} galaxies')

    # Class balance
    n_star_forming = data['labels'].sum().item()
    n_quenched = len(data['labels']) - n_star_forming
    print(
        f'Class balance: {n_quenched:.0f} quenched, '
        f'{n_star_forming:.0f} star-forming'
    )

    # Create model first so we can use its scaling_function
    backbone = torchvision.models.resnet18()
    model = cnn.BernoulliNet(
        lr=lr,
        momentum=0.9,
        backbone=backbone,
        dataset=dataset,
        in_channels=data['X'].shape[1],
    ).to(device)

    # Apply scaling using model's scaling function
    print(f'Scaling data with {model.scaling_function.__name__}')
    data['X'] = model.scaling_function(data['X'])

    # Train/test split - use saved indices if resuming
    if checkpoint is not None and 'train_idxs' in checkpoint:
        print('Using train/test split from checkpoint')
        splits = apply_split_indices(
            data['X'],
            data['rs'],
            data['labels'],
            checkpoint['train_idxs'],
            checkpoint['test_idxs'],
        )
    else:
        splits = train_test_split(
            data['X'],
            data['rs'],
            data['labels'],
            test_fraction=test_fraction,
            seed=seed,
        )
    print(f'Train: {len(splits["X_train"])}, Test: {len(splits["X_test"])}')

    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        splits, batch_size, device
    )

    # Initialize lazy layers with a forward pass
    with torch.no_grad():
        sample_X = splits['X_train'][:2]
        sample_rs = splits['rs_train'][:2]
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
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/f1': train_metrics['f1'],
                'test/loss': test_metrics['loss'],
                'test/accuracy': test_metrics['accuracy'],
                'test/precision': test_metrics['precision'],
                'test/recall': test_metrics['recall'],
                'test/f1': test_metrics['f1'],
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
                splits['train_idxs'],
                splits['test_idxs'],
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
        splits['train_idxs'],
        splits['test_idxs'],
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
        default='gallearn_data_256x256_3proj_wsat_wvmap_avg_sfr_tgt.h5',
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
