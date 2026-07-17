if __name__ == '__main__':
    import argparse

    import gallearn

    parser = argparse.ArgumentParser(
        description=(
            'Record a dataset file\'s sha256 so gallearn.splitting'
            ' and gallearn.train can detect if it silently changes'
            ' under the same filename later'
        )
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=(
            gallearn.config.config['gallearn_paths']['dataset']
        ),
        help='Dataset filename (default: %(default)s)',
    )

    args = parser.parse_args()

    gallearn.dataset_lock.lock_dataset(dataset_fname=args.dataset)
