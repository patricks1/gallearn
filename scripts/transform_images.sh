#!/bin/bash
#SBATCH --partition=ilg2.3
#SBATCH --reservation=cosmo-ilg2.3_hm
#SBATCH --mem=500000
#SBATCH --job-name=standardize_res
#SBATCH -t 2-0:00
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=c-14-35

cd /export/nfs0home/pstaudt/projects/gallearn/scripts
# --threads must match --cpus-per-task above.
julia --project=./ --threads 16 ./transform_images.jl
