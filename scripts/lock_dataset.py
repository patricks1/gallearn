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
    # Positional and required on purpose: a lock file records which
    # dataset a run's results are reproducible against, so defaulting
    # to any particular dataset would let a bare invocation lock one
    # the caller never named.
    parser.add_argument(
        'dataset',
        type=str,
        help='Dataset filename to lock',
    )

    args = parser.parse_args()

    gallearn.dataset_lock.lock_dataset(dataset_fname=args.dataset)
