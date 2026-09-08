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
        default=None,
        help=(
            'Task type: classifier (quenched vs'
            ' star-forming) or regressor (sSFR prediction'
            ' on star-forming galaxies). Required unless'
            ' --resume names a checkpoint that already'
            ' recorded one; must be omitted when --resume'
            ' is given'
        ),
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['standard', 'resnet'],
        default=None,
        help=(
            'Model architecture: standard (torchvision'
            ' ResNet-18 backbone) or resnet (custom'
            ' ResNet). Required unless --resume names a'
            ' checkpoint that already recorded one; must be'
            ' omitted when --resume is given'
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
        '--dataset',
        type=str,
        default=None,
        help=(
            'Train against this dataset instead of the one'
            ' --split records. Use it to run an existing split\'s'
            ' galaxy partition against a dataset holding a'
            ' different target, which keeps results comparable'
            ' across targets and leaves the split file an'
            ' unedited record of its own origin. Must be omitted'
            ' when --resume is given.'
        ),
    )
    parser.add_argument(
        '--target',
        type=str,
        choices=['sfr', 'avg_sfr', 'fgas', 'fdm'],
        default=None,
        help=(
            'Which target the dataset holds. Only for datasets'
            ' built before the pipeline recorded this in the HDF5'
            ' itself; those datasets require it. Passing it for a'
            ' dataset that does declare its own target is an'
            ' error. Must be omitted when --resume is given.'
        ),
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help=(
            'Name for this training run (default: timestamp,'
            ' or a wandb-generated name if --wandb y). Must be'
            ' omitted when --resume is given, since a resumed'
            ' run reuses the checkpoint\'s own recorded'
            ' run_name'
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
        default=None,
        help=(
            'Batch size (default: 32). Must be omitted when'
            ' --resume is given, since a resumed run reuses the'
            ' checkpoint\'s own recorded batch_size'
        ),
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help=(
            'Learning rate (default: 1e-3). Must be omitted when'
            ' --resume is given: a resumed run\'s optimizer state'
            ' already carries whatever lr the scheduler had'
            ' decayed to'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=(
            'Random seed (default: 42). Must be omitted when'
            ' --resume is given, since a resumed run reuses the'
            ' checkpoint\'s own recorded seed and shuffling state'
        ),
    )
    parser.add_argument(
        '-w', '--wandb',
        type=str,
        choices=['n', 'y'],
        default=None,
        help=(
            'Wandb mode: n=none, y=new run (default: n). Must be'
            ' omitted when --resume is given: a resumed run'
            ' automatically continues its checkpoint\'s own wandb'
            ' run if it had one, or continues without wandb'
            ' otherwise'
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
        default=None,
        help=(
            'Disable the learning rate scheduler. Must be omitted'
            ' when --resume is given, since a resumed run reuses'
            ' the checkpoint\'s own recorded use_scheduler'
        ),
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=None,
        help=(
            'Only affects --model standard. Start its ResNet-18'
            ' backbone from ImageNet weights instead of a random'
            ' init. Must be omitted when --resume is given, since'
            ' a resumed run reuses the checkpoint\'s own recorded'
            ' pretrained'
        ),
    )
    parser.add_argument(
        '--head-size',
        type=str,
        choices=sorted(gallearn.train.HEAD_PRESETS),
        default=None,
        help=(
            'Only affects --model resnet: names cnn.ResNet\'s'
            ' fully-connected head shape (default: small). Must be'
            ' omitted when --resume is given, since a resumed run'
            ' reuses the checkpoint\'s own recorded head_size'
        ),
    )

    args = parser.parse_args()

    use_scheduler = (
        None if args.no_scheduler is None else not args.no_scheduler
    )

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
        use_scheduler=use_scheduler,
        pretrained=args.pretrained,
        head_size=args.head_size,
        tgt_type=args.target,
        dataset=args.dataset,
    )
