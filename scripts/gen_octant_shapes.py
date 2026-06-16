import argparse

import gallearn.gen_octant_shapes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Fit Sersic2D profiles to octant galaxy projections and write'
            ' results to a CSV in project_data_dir.'
        ),
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Append to the latest existing output CSV, skipping galaxies'
            ' whose galaxyID already appears in it.'
        ),
    )
    args = parser.parse_args()
    gallearn.gen_octant_shapes.gen(resume=args.resume)
