using CSV
using HDF5
using Printf
using DataFrames
using ProgressBars

direc = "/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/"

function get_N_structures(obj_num)
    obj_num = string(obj_num)
    fname = direc * "ahf_objects_1200/ahf_object_" * obj_num * ".hdf5"
    stars_in_gal = h5open(fname, "r") do file
        read(file, "particleIDs")
    end
    stars_in_gal = Set(stars_in_gal)

    fname = direc * "objects_1200/object_" * obj_num * ".hdf5"
    all_stars, N_structures = h5open(fname, "r") do file
        all_stars = read(file, "stars_id")
        N_structures = read(file, "object_n_substructures")
        return all_stars, N_structures
    end

    in_gal = in.(all_stars, Ref(stars_in_gal))
    N_bound = sum(in_gal)
    N_stars = length(all_stars)
    frac = N_bound / N_stars

    return frac, N_stars, N_bound, N_structures
end

obj_nums = Int[]
N_starss = Int[] 
N_bounds = Int[]
N_structuress = Int[]
fracs = Float64[]

files = readdir(direc * "objects_1200")
for ahf_fname in ProgressBar(files)
    ibeg = findlast('_', ahf_fname) + 1
    iend = findlast(".hdf5", ahf_fname)[1] - 1
    obj_num = ahf_fname[ibeg : iend]
    result = get_N_structures(obj_num)
    append!(obj_nums, parse(Int, obj_num))
    append!(fracs, result[1])
    append!(N_starss, result[2])
    append!(N_bounds, result[3])
    append!(N_structuress, result[4])
end

df = DataFrame(
    obj_num=obj_nums, 
    fracs=fracs, 
    N_stars=N_starss, 
    N_bound=N_bounds,
    N_structures=N_structuress
)

display(df)
CSV.write("objects.csv", df)
