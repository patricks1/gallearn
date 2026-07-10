# Build a "deps-only" Julia system image for the dataset pipeline.
#
# This image bakes GalLearn's and UCITools' heavy third-party
# dependencies (Plots, PyCall, DataFrames, HDF5, Images, Distributions,
# ...) into a .so file, but deliberately leaves GalLearn, UCITools, and
# Revise OUT of the image. Because the two packages you edit are not
# baked, changing their source never invalidates this image: a fresh
# Julia loads all the heavy dependencies from the image instantly and
# only has to compile your own code at runtime.
#
# Run it with the SAME juliaup Julia that runs transform_images.jl, from
# the scripts/ directory:
#
#     ~/.juliaup/bin/julia --project=. build_base_sysimage.jl
#
# transform_images.sh rebuilds the image automatically when it is missing
# or when Manifest.toml is newer than it (a dependency version changed),
# so you rarely run this by hand.

import PackageCompiler

# The packages to bake. This is the union of GalLearn's and UCITools'
# non-stdlib dependencies, minus GalLearn and UCITools themselves (the
# whole point of the image) and minus Revise (kept out so code you are
# actively editing keeps its normal revise-on-load behavior). Every name
# here is a direct dependency of scripts/Project.toml, which PackageCompiler
# requires in order to bake a package.
const BASE_PACKAGES = [
    "BenchmarkTools",
    "CSV",
    "ConfParser",
    "DataFrames",
    "Distributions",
    "FileIO",
    "HDF5",
    "Images",
    "IndexedDataFrames",
    "Interpolations",
    "Pandas",
    "Plots",
    "ProgressBars",
    "PyCall",
    "StatsBase",
]

const OUTPUT = joinpath(@__DIR__, "GalLearn_base_sysimage.so")

# Bake a portable multi-architecture image so one .so runs on every
# partition (ilg2.3, nes2.8, and the login node). Do NOT inherit the
# running Julia's target here: Base.JLOptions().cpu_target is "native",
# which pins the image to the exact CPU of the build node and makes it
# fail to load elsewhere with "Unable to find compatible target in cached
# code image". This is the same multi-target string Julia's own official
# binaries ship with: a generic x86-64 baseline that runs anywhere, plus
# optimized sandybridge and haswell code paths the loader picks when the
# CPU supports them. Set JULIA_CPU_TARGET to override, for example to pin
# the image to one partition's microarchitecture for a bit more speed.
const CPU_TARGET = get(
    ENV,
    "JULIA_CPU_TARGET",
    "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)",
)

println("Building base sysimage:")
println("  output:     $OUTPUT")
println("  cpu_target: $CPU_TARGET")
println("  packages:   ", join(BASE_PACKAGES, ", "))
flush(stdout)

PackageCompiler.create_sysimage(
    BASE_PACKAGES;
    sysimage_path=OUTPUT,
    cpu_target=CPU_TARGET,
)

println("Done. Wrote $OUTPUT")
