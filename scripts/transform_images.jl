using Distributed

# The heavy dependencies (Plots, PyCall, DataFrames, HDF5, ...) come from
# the base sysimage that transform_images.sh loads with -J, so nothing
# below recompiles them. GalLearn and UCITools are not baked into that
# image, so import GalLearn here in the main process first: that triggers
# their precompilation once and writes the cache, and the workers then
# load them from that warm cache instead of each precompiling in parallel.
println("Importing GalLearn..."); flush(stdout)
import GalLearn
println("GalLearn imported."); flush(stdout)

# Spawn one worker per thread. Threads.nthreads() reads the --threads
# flag so the caller controls the worker count from the command line.
# Pass the same base sysimage this process started with to every worker
# (--sysimage) so the workers also skip recompiling the baked
# dependencies. Base.JLOptions().image_file holds this process's image
# path, which is the base sysimage when -J was given and the default
# Julia sysimage otherwise.
nworkers = Threads.nthreads()
sysimage = unsafe_string(Base.JLOptions().image_file)
println("Spawning $nworkers workers..."); flush(stdout)
addprocs(
    nworkers,
    exeflags=[
        "--project=$(dirname(Base.active_project()))",
        "--sysimage=$sysimage",
    ]
)
println("Workers spawned."); flush(stdout)

println("Loading GalLearn on workers..."); flush(stdout)
@everywhere begin
    println("Worker $(myid()) importing GalLearn..."); flush(stdout)
    import GalLearn
    println("Worker $(myid()) done."); flush(stdout)
end
println("GalLearn loaded on workers."); flush(stdout)

GalLearn.Dataset.build_training_data("avg_sfr", Nfiles=nothing, save=true, res=256)
