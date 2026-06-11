using Distributed

# Import GalLearn in the main process first. If precompilation is needed,
# this triggers it once and warms the cache so workers load instantly.
println("Importing GalLearn..."); flush(stdout)
import GalLearn
println("GalLearn imported."); flush(stdout)

# Spawn one worker per thread. Threads.nthreads() reads the --threads
# flag so the caller controls the worker count from the command line.
nworkers = Threads.nthreads()
println("Spawning $nworkers workers..."); flush(stdout)
addprocs(
    nworkers,
    exeflags="--project=$(dirname(Base.active_project()))"
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
