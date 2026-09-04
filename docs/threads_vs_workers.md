# Threads, workers, and the heap

A reference for the parallelism in `src/Dataset.jl`. It explains what
threads and workers actually are, which one `pmap` uses, and why the
dataset build uses both instead of picking one.

## The one-sentence version

A **worker** is a whole separate program with its own memory, and a
**thread** is an extra worker *inside* one program sharing that
program's memory. `pmap` uses workers. `Threads.@threads` uses
threads.

## Process, heap, thread, worker

- A **process** is an independent OS-level instance of a running
  program: its own virtual address space, its own copy of every
  loaded library. Nothing in one process's memory is visible to
  another process unless something explicitly arranges it (a pipe,
  shared memory, sockets). The Julia session you start is one
  process; each worker `addprocs` spawns is another.
- The **heap** is the region of a process's memory holding
  dynamically allocated objects (arrays, `Dict`s, `DataFrame`s) for
  as long as something still references them. It exists per
  process. One process cannot read or write another process's heap
  directly; getting data from one to another means copying it.
- A **thread** is a unit of execution inside one process, scheduled
  by the OS, sharing that process's heap and everything else about
  it. Spawning one costs a stack (a few MB) and nothing else, since
  it reuses the process's already-loaded code and data.
- A **worker**, in Julia's `Distributed` sense, is a whole separate
  OS process, spawned by `addprocs` and controlled from the calling
  ("main") process over a socket. It starts its own Julia runtime
  and loads its own copy of every package the main process has
  loaded, on its own heap. That separateness is why each one costs
  roughly 2 GB here before it has touched any data: a full Julia
  runtime, `HDF5`, `CSV`, `DataFrames`, `PyCall` (which also starts
  an embedded Python interpreter), and whatever native code gets
  JIT-compiled for the functions it runs.

| Term | How you get it | What it costs |
|---|---|---|
| Process | `addprocs(n)` spawns one per worker | Full Julia runtime + every loaded package, ~2 GB each here |
| Heap | Automatic, one per process | The memory your arrays actually occupy |
| Thread | `julia --threads n` | A stack, a few MB; shares the process's heap |
| Serialization | Automatic, when a worker returns a result | Time, plus a second copy of the data |

Two consequences follow, and they drive every decision below.

1. **Threads can write into a shared array. Workers cannot.** Ten
   threads each filling a different row of one big array are all
   writing into the same heap, so the array is simply finished when
   they are done. Ten workers cannot do that. Each builds its own
   piece on its own heap, ships it back over its socket, and the
   main process copies it into place. For a moment both the shipped
   pieces and the final array exist, so peak memory is roughly
   double, unless something drops the shipped copies once they stop
   being needed. The case study near the end of this document is a
   real instance of that "unless."

2. **Workers give you real parallel file reading. Threads sometimes
   do not.** More on this next.

## Which one does `pmap` use?

`pmap` uses **workers** (processes). So do `@distributed` and
`remotecall`. They all come from the `Distributed` standard library,
and they all imply separate memory.

`Threads.@threads` uses **threads**. It comes from `Base.Threads`,
and it implies shared memory.

The two are unrelated knobs. Setting one does not set the other.

## Why this pipeline uses both

`src/Dataset.jl` reads image files with `pmap` (workers) and fills
the velocity-map channel with `Threads.@threads` (threads). That
looks inconsistent until you see what each phase needs.

### Reading images uses workers

Pass 1 and pass 2 open thousands of HDF5 files. HDF5 is thread-safe
here (`HDF5.API.h5_is_library_threadsafe()` returns `true`), but it
achieves that with a single lock per process. Ten threads in one
process all wanting to open an HDF5 file queue on that one lock, one
at a time, so ten threads reading HDF5 do not read ten times faster.

Ten worker processes each hold their own lock. Ten worker processes
genuinely read ten files at once. That is why the read-heavy passes
pay for workers.

The price is serialization. Every image a worker reads has to be
serialized, shipped to the main process, and copied into `X`.

### Filling the velocity maps uses threads

By this point `X` already exists in the main process and is about
15 GiB. The work left is to write one more channel into rows of that
existing array.

Workers would be a poor fit here. Each would need its slice of `X`
shipped out and shipped back, moving 15 GiB across process
boundaries to avoid a copy that costs nothing when done in place.
Threads just write into the array. `load_vmap` still opens an HDF5
file per galaxy, so those reads, decompression included, queue on
the one global lock, one thread at a time. Only the work after the
read, the `permutedims` and the write into `X`, runs in parallel.
The reads are small, so the lock is not the bottleneck it would be
for the image-loading passes, but it is not free parallelism either.

So the rule this codebase follows is:

> Use workers when parallel **reading** dominates. Use threads when
> writing into one big shared array dominates.

## Tasks, `@async`, and `fetch`

`pmap` blocks by default. Called plainly, it would not return until
every file in pass 1 had been scanned, so nothing else, including
the progress bar, could run in the meantime. `Dataset.jl` gets
around that with a `Task`.

```julia
p1_task = @async Distributed.pmap(good_paths[1:Nfiles]) do path
    result = scan_file(path)
    put!(prog1, nothing)
    result
end
```

`@async` and `fetch` both come from `Base`, not from `Distributed`.
They belong to Julia's task system, a third mechanism alongside
threads and workers. A **`Task`** is a unit of work Julia can pause
and resume. `@async expr` wraps `expr` in a `Task`, schedules it to
run, and immediately hands back the `Task` object itself, not its
result. The calling code then carries on to whatever comes next
while that `Task` runs.

Here, what comes next is the loop that drains `prog1`:

```julia
valid_projs_per_file = begin
    n = 0
    while n < Nfiles
        take!(prog1)
        n += 1
        tick_progress(n, Nfiles, "Pass 1", t_p1)
    end
    println()
    fetch(p1_task)
end
```

`prog1` is a `Channel{Nothing}`, made a `RemoteChannel` so workers
can reach it too. Every time a worker's `scan_file` call finishes,
that worker calls `put!(prog1, nothing)`, one `nothing` per finished
file. The value carries no information; it exists purely as a
token, one per completed file. `take!(prog1)` removes one token, and
`while n < Nfiles` keeps taking until it has removed all of them.
Since a token only appears after its file is done, that loop cannot
exit until every file has been scanned. It is what actually
synchronizes on the `pmap` call finishing, and it draws the
progress bar as a side effect of doing so.

By the time that loop exits, `pmap` has already assembled its
result, an ordered `Vector` matching `good_paths[1:Nfiles]`
regardless of which worker finished which file when. `fetch(p1_task)`
retrieves that value from the `Task`. Picture the `Task` as a box
that eventually holds its result. `@async` hands you the box before
anything is inside it, the work fills the box while it runs
concurrently with the `while` loop, and `fetch` reads what is
inside. If the box were not already full, `fetch` would block until
it was; here it returns immediately, since the `while` loop already
waited. `fetch` also rethrows any exception the task raised, instead
of losing it silently.

The serializing and copying that moves a worker's result back to
the main process is `pmap`'s work, done while the `Task` runs, not
something `fetch` performs. `fetch` only reads the finished value
back out.

Pass 2 (`p2_task`, `prog2`) follows the identical pattern for image
loading.

## How the numbers get set here

This is where the naming actively misleads. Follow the chain:

1. The sbatch wrapper runs
   `julia --threads $NWORKERS ./build_dataset.jl`, so the main
   process starts with `NWORKERS` **threads**.
2. `build_dataset.jl` then does `nworkers = Threads.nthreads()`,
   reading that thread count back.
3. It calls `addprocs(nworkers)`, spawning that many **worker
   processes**.

With `NWORKERS=16` you get **17 processes**: one main process
holding `X` and running 16 threads, plus 16 worker processes with
one thread each.

Nothing requires those two numbers to match. Line 2 is the only
thing tying them together. You could run 16 threads and 4 workers.
The variable named `nworkers` holding a thread count is why this
reads as one setting when it is really two.

It matters because the two cost completely different amounts. The 16
threads are close to free. The 16 workers cost roughly 32 GB. On a
node with 47 GB, lowering the worker count is the memory lever, and
lowering the thread count only makes the velocity-map and assembly
loops slower for no saving.

## Do threads and workers map onto hardware?

Not with a fixed binding. A **hardware thread** is a logical core.
A chip with hyperthreading exposes two hardware threads per physical
core, so a 16-physical-core node can expose 32 hardware threads.

A Julia **thread** is an OS thread (a pthread on Linux). The kernel
scheduler maps each OS thread onto whichever hardware thread is free
at a given instant, and it can move that OS thread to a different
hardware thread later. Nothing pins a Julia thread to one core unless
you set CPU affinity yourself.

A **worker** is a whole OS process. By default each worker process
runs single-threaded, so while it is active it occupies one hardware
thread the same way one Julia thread does. The kernel schedules it
the same way too.

So both threads and workers ultimately run on hardware threads, but
through one extra layer (the OS scheduler) rather than a direct
assignment. That layer is also where oversubscription shows up.
Asking for more Julia threads, or more workers, than the node has
hardware threads does not create more parallelism. The kernel just
time-slices the excess ones, and the ones sharing a hardware thread
stop running genuinely simultaneously.

This is why sizing matters. `--threads n` should match the hardware
threads Slurm actually allocated to the job (its `--cpus-per-task`),
not some larger number picked for its own sake. Workers add on top
of that. `build_dataset.jl` sets `nworkers = Threads.nthreads()`
(line 22 as of this writing), so during any phase where the main
process's threads and its 16 worker processes are both alive, the
job asks for roughly double the hardware threads it was allocated.

## Case study: reachable-but-dead memory in `load_images`

Pass 2's assembly loop copies `results` (the `Vector` that
`fetch(p2_task)` returns) row by row into `X`. Once that copy
finishes, nothing in `load_images` reads `results` again, but the
local variable is still bound to it. That matters because the
garbage collector frees a value only once nothing reachable still
points to it, and a live local variable is exactly such a pointer.
Being unused is not the same as being unreachable.

Measuring `Base.gc_live_bytes()` at 7,740 rows (half the usual
dataset size) showed exactly that gap. Forcing `GC.gc()` right after
the assembly loop, with `results` still bound, changed nothing:
13.39 GiB live before, 13.33 GiB after. Setting `results = nothing`
first, then forcing `GC.gc()`, dropped live memory to 7.69 GiB,
5.6 GiB freed by that one assignment. `load_images` runs several
more allocating steps after the assembly loop and before its
`return`, mask-building, the `Re` lookup, and a full second copy of
`X` if `logandscale` is on, all of which had been paying rent on
`results`'s dead weight the whole time.

The fix is one line, `results = nothing` right after the assembly
loop, followed by a `GC.gc()` call so the collector reclaims it
there rather than whenever it next happens to run on its own. The
general lesson extends past this one function. Reachability, not
usefulness, is what keeps the garbage collector from freeing
something. A large local variable you are done with still costs
memory until it goes out of scope or gets explicitly cleared.

## Quick reference

| Question | Threads | Workers |
|---|---|---|
| Separate memory? | No, one shared heap | Yes, one heap each |
| Can write into `X` directly? | Yes | No, results get copied back |
| Cost to add one | Negligible | About 2 GB |
| Created by | `--threads n` | `addprocs(n)` |
| Used by | `Threads.@threads` | `pmap`, `@distributed` |
| Parallel HDF5 reads? | No, one lock per process | Yes, one lock each |
| Removed by | Ending the program | `rmprocs(workers())` |

## Common confusions

**"`rmprocs` killed the workers, so the threads are gone too."** No.
They are different things. `rmprocs(workers())` shuts down the
worker processes. The main process and its threads keep running,
which is exactly why `src/Dataset.jl` can free the workers right
after the reading passes and still run the velocity-map loop in
parallel.

**"More workers is always faster."** Each one costs about 2 GB. Past
the point where they stop fitting in RAM the node starts swapping and
everything becomes far slower than running fewer.

**"The garbage collector frees memory, so peak memory does not
matter."** The collector frees memory once nothing refers to it, but
peak is what has to fit. If two 15 GiB arrays exist at the same
moment, the node needs 30 GiB at that moment, no matter how quickly
one is freed afterward. Julia also does not always hand freed pages
back to the operating system right away, so the process can look
large after the collector has run. And "nothing refers to it" is a
stricter condition than "nothing uses it". The case study above is a
value nobody used again that still blocked collection, because a
live variable still referred to it.

**"Threads are just lightweight workers."** They are lighter, but the
real difference is shared memory versus separate memory, and that
difference decides which one a given phase can even use.
