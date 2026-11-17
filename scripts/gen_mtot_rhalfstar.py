'''
Generate the mtot_rhalfstar target, each FIREBox galaxy's total mass (stars +
gas + dark matter) within its 3D half-stellar-mass radius. The computation
lives in uci_tools.process_firebox; this script runs it over the whole sample
and saves mtot_rhalfstar.csv to project_data_dir.
'''
import argparse

from uci_tools import process_firebox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--nworkers',
        type=int,
        default=None,
        help='Number of worker processes. Default uses every allocated core.',
    )
    args = parser.parse_args()
    process_firebox.get_all_mtot_rhalfstar(
        save=True,
        nworkers=args.nworkers,
    )


if __name__ == '__main__':
    main()
