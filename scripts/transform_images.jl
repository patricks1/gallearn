using Distributed

# Import GalLearn in the main process first. If precompilation is needed,
# this triggers it once and warms the cache so workers load instantly.
import GalLearn

# Spawn one worker per thread. Threads.nthreads() reads the --threads
# flag so the caller controls the worker count from the command line.
nworkers = Threads.nthreads()
addprocs(
    nworkers,
    exeflags="--project=$(dirname(Base.active_project()))"
)

@everywhere import GalLearn

GalLearn.Dataset.build_training_data("avg_sfr", Nfiles=nothing, save=true, res=256)
