if __name__ == '__main__':
    import argparse
    import gallearn

    parser = argparse.ArgumentParser(description='Run the galaxy shape CNN.')
    parser.add_argument(
        '-n',
        '--num-gals',
        type=int,
        required=False, 
        help=(
            'The number of galaxies to use in the network. If not specified,'
            ' the network will use all galaxies available.'
        )
    )
    parser.add_argument(
        '-w',
        '--wandb',
        type=str,
        choices=['n', 'y', 'r'],
        default='n',
        help=(
            '`wandb` mode. Choices are'
            ' \'n\': No'
            ' interaction.'
            ' \'y\': Yes, start a new run.'
            ' \'r\': Resume a run.'
        )
    )
    parser.add_argument(
        '-r',
        '--run-name',
        type=str,
        help=(
            'The name to give a new run, or the name of the run to resume.'
            ' (Required'
            ' if --wandb is \'r\')'
        )
    )

    args = parser.parse_args()
    if args.wandb == 'r' and args.run_name is None:
        parser.error('--run-name is required when --wandb is \'r\'')

    Nfiles = args.num_gals
    wandb_mode = args.wandb
    run_name = args.run_name
    gallearn.cnn.main(Nfiles, wandb_mode=wandb_mode, run_name=run_name)
