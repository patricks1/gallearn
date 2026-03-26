module Dataset

using HDF5
using CSV
using DataFrames
import Images
import StatsBase
import Plots
import ..GalLearnConfig

conf = GalLearnConfig.read_config()

#sat_direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/satellite/" *
#    "ugrband_massmocks_final"
sat_direc = conf["gallearn_paths"]["sat_image_dir"]

#host_direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/host/" *
#    "ugrband_massmocks_final"
host_direc = conf["gallearn_paths"]["host_image_dir"]
octant_img_dir = conf["gallearn_paths"]["octant_img_dir"]

gallearn_dir = conf["gallearn_paths"]["project_data_dir"]
tgt_3d_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn/luke_protodata"
tgt_sfr_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn/"
tgt_2d_host_path = conf["gallearn_paths"]["host_2d_shapes"]
tgt_2d_sat_path = conf["gallearn_paths"]["sat_2d_shapes"]
output_dir = conf["gallearn_paths"]["project_data_dir"]

function fmt_time(secs)
    m = floor(Int, secs / 60)
    s = floor(Int, secs % 60)
    return "$(m)m $(s)s"
end

# tick_progress! increments a shared atomic counter and prints a progress
# line whenever the integer percentage ticks up. Call once per completed
# work item at the end of a Threads.@threads loop body. t_start is the
# value of time() captured before the loop.
function tick_progress!(counter, total, label, t_start)
    prev = Threads.atomic_add!(counter, 1)
    pct_before = (prev * 100) ÷ total
    pct_after = ((prev + 1) * 100) ÷ total
    if pct_after > pct_before
        elapsed = time() - t_start
        remaining = elapsed * (total - (prev + 1)) / (prev + 1)
        print(
            "\r  $(label): $(pct_after)% " *
            "($(fmt_time(elapsed)) elapsed, " *
            "$(fmt_time(remaining)) remaining)"
        )
    end
end

# scan_file reads only HDF5 dataset-size metadata (not pixel data) for one
# file and returns the projection keys whose images have width <= 2000.
function scan_file(path)
    valid_projs = String[]
    HDF5.h5open(path, "r") do file
        proj_keys = sort(filter(
            k -> startswith(k, "projection_"), keys(file)
        ))
        for proj in proj_keys
            # size() on an HDF5Dataset reads only the dataspace metadata,
            # not the pixel data, so this pass is cheap.
            sz = size(file[proj * "/band_r"])
            if sz[end] <= 2000
                push!(valid_projs, proj)
            end
        end
    end
    return valid_projs
end

function read_2d_tgt()
    function csv_read(tgt_path)
        # Header for host file is
        #     galaxyID
        #     FOV
        #     pixel
        #     view
        #     band
        #     b_a
        #     PA
        #     n
        #     Re
        #     Ie
        # Header for the satellite file is
        #     galaxyID
        #     FOV
        #     pixel
        #     view
        #     band
        #     b_a
        #     PA
        #     n
        #     Re
        #     Ie
        #     Mstar_ahf_cat
        #     flag
        dat = CSV.read(
            tgt_path,
            DataFrame,
            header=1
        )
        return dat
    end
    host_dat = csv_read(tgt_2d_host_path)
    sat_dat = csv_read(tgt_2d_sat_path)
    # Remove the Mstar_ahf_cat and flag columns from sat_dat because those
    # columns aren't in host_dat, and we need to combine the two.
    DataFrames.select!(sat_dat, DataFrames.Not([:Mstar_ahf_cat, :flag]))
    dat = vcat(host_dat, sat_dat)

    dat.galaxyID .= "object_" .* string.(dat.galaxyID)
    DataFrames.rename!(dat, :galaxyID => :Simulation)

    dat = dat[dat.band .== "band_r", :]

    return dat
end

function read_3d_tgt()
    files = readdir(tgt_3d_dir)
    ys = CSV.read(joinpath(tgt_3d_dir, "FIREBoxm9.csv"), DataFrame)
    for mclass in ["7", "8", "10"]
        ys_add = CSV.read(
            joinpath(tgt_3d_dir, "FIREBoxm" * mclass * ".csv"),
            DataFrame
        )
        ys = vcat(ys, ys_add)
    end
    return ys
end

function read_sfr_tgt(sfr_type)
    if sfr_type == "sfr"
        fname = "sfrs.csv"
    elseif sfr_type == "avg_sfr"
        fname = "avg_sfrs.csv"
    end
    y_df = CSV.read(joinpath(tgt_sfr_dir, fname), DataFrame)
    y_df.id .= "object_" .* string.(y_df.id)
    DataFrames.rename!(y_df, :id => :Simulation)
    return y_df
end

function load_images(
            ; Nfiles=nothing,
            logandscale=false,
            res=256,
            tgt_type="2d"
        )
    host_fnames = filter(
        f -> isfile(joinpath(host_direc, f)) && endswith(f, ".hdf5"),
        readdir(host_direc)
    )
    host_paths = joinpath.(host_direc, host_fnames)
    sat_fnames = filter(
        f -> isfile(joinpath(sat_direc, f)) && endswith(f, ".hdf5"),
        readdir(sat_direc)
    )
    sat_paths = joinpath.(sat_direc, sat_fnames)
    octant_fnames = filter(
        f -> isfile(joinpath(octant_img_dir, f)) && endswith(f, ".hdf5"),
        readdir(octant_img_dir)
    )
    octant_paths = joinpath.(octant_img_dir, octant_fnames)
    files = [host_fnames; sat_fnames; octant_fnames]
    paths = [host_paths; sat_paths; octant_paths]

    baddies = [
        "object_1162",
        "object_280",
    ]
    is_bad = [any(occursin(baddy, f) for baddy in baddies) for f in files]

    if tgt_type == "3d"
        y_df = read_3d_tgt()
        all_bands = true
        Nbands = 3
    elseif tgt_type == "2d"
        y_df = read_2d_tgt()
        all_bands = false
        Nbands = 1
    elseif tgt_type in ("sfr", "avg_sfr")
        y_df = read_sfr_tgt(tgt_type)
        all_bands = true
        Nbands = 3
    else
        throw(ArgumentError(
            "`tgt_type` should be \"2d\", \"3d\", \"sfr\", or \"avg_sfr\"."
        ))
    end

    # For every file name, create an inner mask the size of y_df.Simulation
    # where a `true` marks the simulation in y_df (if any) whose name occurs
    # in that file name. If any row in that inner mask is true (although
    # there should be at most one), mark the file with a `true` in the
    # `in_tgt` outer mask.
    in_tgt = [
        any(occursin(obj * "_", f) for obj in y_df.Simulation)
        for f in files
    ]

    good_files = files[.!is_bad .& in_tgt]
    good_paths = paths[.!is_bad .& in_tgt]
    if Nfiles === nothing
        # If the user hasn't specified the number of files to run through,
        # run through all of them.
        Nfiles = length(good_files)
    else
        good_indices = StatsBase.sample(
            1:length(good_files),
            Nfiles,
            replace=false
        )
        good_files = good_files[good_indices]
        good_paths = good_paths[good_indices]
    end

    # Pass 1: scan each file for valid (non-oversized) projection keys.
    # Each thread works on a separate file, so no locking is needed.
    # scan_file reads only HDF5 metadata, so this pass is fast.
    println(
        "Pass 1: scanning $(Nfiles) files for valid projections " *
        "using $(Threads.nthreads()) threads..."
    )
    valid_projs_per_file = Vector{Vector{String}}(undef, Nfiles)
    p1_done = Threads.Atomic{Int}(0)
    p1_start = time()
    Threads.@threads for fi in 1:Nfiles
        valid_projs_per_file[fi] = scan_file(good_paths[fi])
        tick_progress!(p1_done, Nfiles, "Pass 1", p1_start)
    end
    println() # Advance past the \r progress line.

    # Compute per-file starting row indices via prefix sum so that pass 2
    # can write each file's rows without any shared mutable state.
    counts = length.(valid_projs_per_file)
    offsets = cumsum([1; counts[1:end-1]])
    N_rows = sum(counts)

    X = zeros(N_rows, Nbands, res, res)
    ids_X = Vector{String}(undef, N_rows)
    fnames_sorted = Vector{String}(undef, N_rows)
    orientations = Vector{String}(undef, N_rows)

    # Pass 2: load and resize pixel data in parallel. Each thread fills its
    # pre-assigned rows in X, ids_X, fnames_sorted, and orientations, so
    # no locking is needed. Same atomic progress counter as pass 1.
    println("Pass 2: loading images...")
    p2_done = Threads.Atomic{Int}(0)
    p2_start = time()
    Threads.@threads for fi in 1:Nfiles
        fname = good_files[fi]
        path = good_paths[fi]
        projs = valid_projs_per_file[fi]
        base_row = offsets[fi]
        underscores = findall(isequal('_'), fname)
        obj_id = fname[1:underscores[2]-1]
        HDF5.h5open(path, "r") do file
            for (pi, proj) in enumerate(projs)
                row = base_row + pi - 1
                bands_list = all_bands ? ["g", "u", "r"] : ["r"]
                img = zeros(Nbands, res, res)
                for (bi, band) in enumerate(bands_list)
                    band_data = read(file, proj * "/band_" * band)
                    if size(band_data) != (res, res)
                        band_data = Images.imresize(band_data, res, res)
                    end
                    img[bi, :, :] = parent(band_data)
                end
                X[row, :, :, :] = img
                ids_X[row] = obj_id
                orientations[row] = proj
                fnames_sorted[row] = fname
            end
        end
        tick_progress!(p2_done, Nfiles, "Pass 2", p2_start)
    end
    println() # Advance past the \r progress line.
    println("Done loading images. X shape: $(size(X))")

    if logandscale
        # Turn zeros into the smallest value greater than zero to avoid
        # -Inf in the logs.
        is_zero = X .== 0
        X[is_zero] .= minimum(X[.!is_zero])
        logX = log10.(X)
        X = (
            (logX .- minimum(logX)) ./ (maximum(logX) .- minimum(logX))
        )
    end

    if tgt_type in ["2d", "3d"]
        id_mask = falses(length(ids_X), size(y_df)[1])
        orientation_mask = copy(id_mask)
        for i in 1:length(ids_X)
            id_mask[i, :] .= y_df[!, "Simulation"] .== ids_X[i]
            orientation_mask[i, :] .= y_df[!, "view"] .== orientations[i]
        end
        mask = id_mask .& orientation_mask
    elseif tgt_type in ("sfr", "avg_sfr")
        mask = falses(length(ids_X), size(y_df)[1])
        for i in 1:length(ids_X)
            # For every id_X, generate a vector of booleans corresponding
            # to the rows in y_df where Simulation matches id_X. There
            # should only be one match for each id_X.
            mask[i, :] .= y_df[!, "Simulation"] .== ids_X[i]
        end
    else
        throw(ArgumentError(
            "`tgt_type` should be \"2d\", \"3d\", \"sfr\", or \"avg_sfr\"."
        ))
    end

    # Ensure there's only one match for every Xi.
    @assert all(sum(mask[i, :]) == 1 for i in 1:size(mask, 1))
    indices = findfirst.(eachrow(mask))
    y_df = y_df[indices, :]
    if tgt_type in ["2d", "3d"]
        # Ensure the orientations match between X and y_df.
        @assert all(orientations .== y_df[:, "view"])
    end
    # Ensure the galaxies match between X and y_df
    @assert all(ids_X .== y_df[:, "Simulation"])

    return ids_X, X, fnames_sorted, y_df, orientations
end

function load_vmap(id, res)
    maps_dir = conf["gallearn_paths"]["vmaps_dir"]
    path = joinpath(maps_dir, "object_$(id)_vmap.hdf5")
    vmap = Dict{String, Dict}()
    if !isfile(path)
        return nothing
    end
    HDF5.h5open(path, "r") do file
        for orientation in keys(file)
            for data in keys(file[orientation])
                vmap[orientation] = read(file, orientation)
            end
        end
    end
    return vmap
end

function build_training_data(tgt_type; Nfiles=nothing, save=false, res=256)
    ids_X, X, files, y_df, orientations_X = load_images(
        Nfiles=Nfiles,
        res=res,
        tgt_type=tgt_type
    )
    # Add in velocity map.
    VMAP = zeros(length(ids_X), 1, res, res)
    orientations = unique(orientations_X)
    ids_set = unique(ids_X)

    # Preflight check: before doing any heavy work, scan the HDF5 keys of
    # every vmap file and verify that all orientations present in the image
    # data for that galaxy are also present in its vmap. This catches
    # image/vmap mismatches at the start of the run rather than deep in the
    # loop below.
    maps_dir = conf["gallearn_paths"]["vmaps_dir"]
    mismatches = Dict{String, Vector{String}}()
    # mismatch_lock protects mismatches since multiple threads may find
    # problems and write to it concurrently.
    mismatch_lock = ReentrantLock()
    n_ids = length(ids_set)
    preflight_done = Threads.Atomic{Int}(0)
    preflight_start = time()
    Threads.@threads for id in ids_set
        id_int = parse(Int, replace(id, "object_" => ""))
        vmap_path = joinpath(
            maps_dir,
            "object_$(id_int)_vmap.hdf5"
        )
        if isfile(vmap_path)
            # Read only the top-level keys -- no pixel data loaded.
            vmap_keys = HDF5.h5open(vmap_path, "r") do f
                keys(f)
            end
            id_orients = unique(orientations_X[ids_X .== id])
            missing_orients = setdiff(id_orients, vmap_keys)
            if !isempty(missing_orients)
                lock(mismatch_lock) do
                    mismatches[id] = collect(missing_orients)
                end
            end
        end
        tick_progress!(preflight_done, n_ids, "Preflight", preflight_start)
    end
    println() # Advance past the \r progress line.
    if !isempty(mismatches)
        println(
            "Preflight: $(length(mismatches)) galaxies have image " *
            "orientations missing from their vmaps:"
        )
        for (id, missing_orients) in sort(collect(mismatches))
            println("  $id: $missing_orients")
        end
        error("Orientation mismatch between images and vmaps.")
    end

    # Each galaxy reads its own vmap file and writes to non-overlapping rows
    # of VMAP (each (id, orientation) pair maps to a unique row), so
    # @threads requires no locking here.
    println("Loading vmaps...")
    vmap_done = Threads.Atomic{Int}(0)
    vmap_start = time()
    Threads.@threads for id in ids_set
        id_int = parse(Int, replace(id, "object_" => ""))
        vmap = load_vmap(id_int, res)
        if !isnothing(vmap)
            for orientation in orientations
                is_id = ids_X .== id
                is_orient = orientations_X .== orientation
                is_i = is_id .& is_orient
                i = findall(is_i)
                @assert length(i) == 1
                i = i[1]
                VMAP[i, 1, :, :] = vmap[orientation]["vmap"]
            end
        end
        tick_progress!(vmap_done, n_ids, "Loading vmaps", vmap_start)
    end
    println() # Advance past the \r progress line.
    X  = cat(X, VMAP; dims=2)
    println("X shape: $(size(X))")

    if tgt_type == "3d"
        ys = Array(y_df[:, ["b/a", "c/a"]])
    elseif tgt_type == "2d"
        ys = Array(y_df[:, "b_a"])
        println("`ys` type: " * string(typeof(ys)))
        ys = reshape(ys, (size(ys)..., 1))
        println("`ys` shape: " * string(size(ys)))
    elseif tgt_type in ("sfr", "avg_sfr")
        ys = Array(y_df[:, "ssfr"])
        ys = reshape(ys, (size(ys)..., 1))
    else
        throw(ArgumentError(
            "`tgt_type` should be \"2d\", \"3d\", \"sfr\", or\"avg_sfr\"."
        ))
    end

    if save
        fname = "gallearn_data_" *
            string(res) *
            "x" *
            string(res) *
            "_11proj_wsat_wvmap"

        # Sample type
        if Nfiles !== nothing
            fname *= "_" * string(Nfiles) * "gal_subsample"
        end

        # 2d, 3d, sfr, or avg_sfr target data
        fname *= "_" * tgt_type * "_tgt"

        # Temporary because I'm testing multiprocessing while a serialized run
        # is already going.
        fname *= "_mp" 

        fname *= ".h5"

        h5open(joinpath(output_dir, fname), "w") do f
            # Permute X and ys before writing so that h5py reads them
            # in PyTorch (row-major) order. Julia writes column-major
            # data to HDF5, and h5py reverses all axes on read. Writing
            # X as (W, H, C, N) means h5py reads (N, C, H, W). Writing
            # ys as (features, N) means h5py reads (N, features).
            X_out = permutedims(X, (4, 3, 2, 1))
            ys_out = permutedims(ys, (2, 1))
            for (label, data) in [
                        ["X", X_out],
                        ["obs_sorted", ids_X],
                        ["orientations", orientations_X],
                        ["file_names", files],
                        ["ys_sorted", ys_out],
                    ]
                println(
                    "Saving "
                    * label
                    * " of type $(typeof(data))"
                )
                write(f, label, data)
            end
        end
    end

    return ids_X, y_df, ys, X, files, orientations_X
end

end # module Dataset
