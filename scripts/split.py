if __name__ == '__main__':
    import argparse

    import gallearn

    parser = argparse.ArgumentParser(
        description=(
            'Create and update galaxy train/val/test splits'
        )
    )
    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
    )

    lock_parser = subparsers.add_parser(
        'test-lock',
        help='Create or top up the locked test set',
    )
    lock_parser.add_argument(
        '--dataset',
        type=str,
        default=(
            gallearn.config.config['gallearn_paths']['dataset']
        ),
        help='Dataset filename (default: %(default)s)',
    )
    lock_parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.10,
        help=(
            'Target locked share of each mass/sSFR stratum'
            ' (default: %(default)s)'
        ),
    )
    lock_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help=(
            'Random seed for stratified sampling'
            ' (default: %(default)s)'
        ),
    )
    lock_parser.add_argument(
        '--n-mass-bins',
        type=int,
        default=5,
        help=(
            'Quantile bins on log10(Mstar) (default: %(default)s)'
        ),
    )
    lock_parser.add_argument(
        '--n-ssfr-bins',
        type=int,
        default=5,
        help=(
            'Quantile bins on log10(ssfr) (default: %(default)s)'
        ),
    )

    split_parser = subparsers.add_parser(
        'split',
        help='Create a train/val split from the unlocked galaxies',
    )
    split_parser.add_argument(
        '--dataset',
        type=str,
        default=(
            gallearn.config.config['gallearn_paths']['dataset']
        ),
        help='Dataset filename (default: %(default)s)',
    )
    split_parser.add_argument(
        '--test-lock',
        type=str,
        default=None,
        help=(
            'Path to a specific test_lock_v<N>.json. Defaults to'
            ' the highest version in'
            ' gallearn.splitting.SPLITS_DIR.'
        ),
    )
    split_parser.add_argument(
        '--val-fraction',
        type=float,
        default=0.15,
        help=(
            'Target val share of the unlocked galaxies'
            ' (default: %(default)s)'
        ),
    )
    split_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help=(
            'Random seed for the train/val shuffle'
            ' (default: %(default)s)'
        ),
    )
    split_parser.add_argument(
        '--split-name',
        type=str,
        default=None,
        help=(
            'Used in the output filename'
            ' (SPLITS_DIR/split_<split-name>.json). Defaults to'
            ' "<dataset stem>_v<N>", N one more than the highest'
            ' existing split version for this dataset'
        ),
    )

    args = parser.parse_args()

    if args.command == 'test-lock':
        gallearn.splitting.update_test_lock(
            dataset_fname=args.dataset,
            target_fraction=args.test_fraction,
            seed=args.seed,
            n_mass_bins=args.n_mass_bins,
            n_ssfr_bins=args.n_ssfr_bins,
        )
    elif args.command == 'split':
        gallearn.splitting.write_split(
            dataset_fname=args.dataset,
            test_lock_path=args.test_lock,
            split_name=args.split_name,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
