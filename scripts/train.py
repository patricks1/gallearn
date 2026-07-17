if __name__ == '__main__':
    import argparse

    import gallearn

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
        '--split',
        type=str,
        default=None,
        help=(
            'Path to a train/val split JSON, from'
            ' scripts/split.py split. Its recorded dataset_path'
            ' determines the dataset this run trains against.'
            ' Required unless --resume names a checkpoint that'
            ' already recorded one; must be omitted when --resume'
            ' is given.'
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

    gallearn.train.main(
        task=args.task,
        model_type=args.model,
        split_file_path=args.split,
        run_name=args.run_name,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        wandb_mode=args.wandb,
        resume_from=args.resume,
        use_scheduler=not args.no_scheduler,
    )
