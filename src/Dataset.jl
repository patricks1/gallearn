
module Dataset

using HDF5
using CSV
using DataFrames
using Distributed
import Images
import StatsBase
import Plots
import ..GalLearnConfig

# Module-level path variables populated by __init__ at load time (not
# precompile time) so that Pkg.API.project() resolves correctly.
sat_direc = ""
host_direc = ""
octant_img_dir = ""
gallearn_dir = ""
tgt_3d_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn/luke_protodata"
tgt_sfr_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn/"
host_2d_shapes_path = ""
sat_2d_shapes_path = ""
octant_2d_shapes_path = ""
output_dir = ""
maps_dir = ""

function __init__()
    conf = GalLearnConfig.read_config()
    global sat_direc = conf["gallearn_paths"]["sat_image_dir"]
    global host_direc = conf["gallearn_paths"]["host_image_dir"]
    global octant_img_dir = conf["gallearn_paths"]["octant_img_dir"]
    global gallearn_dir = conf["gallearn_paths"]["project_data_dir"]
    global host_2d_shapes_path = conf["gallearn_paths"]["host_2d_shapes"]
    global sat_2d_shapes_path = conf["gallearn_paths"]["sat_2d_shapes"]
    global octant_2d_shapes_path = conf["gallearn_paths"]["octant_shapes"]
    global output_dir = conf["gallearn_paths"]["project_data_dir"]
    global maps_dir = conf["gallearn_paths"]["vmaps_dir"]
end

function fmt_time(secs)
    m = floor(Int, secs / 60)
    s = floor(Int, secs % 60)
    return "$(m)m $(s)s"
end

# tick_progress prints a \r progress line whenever the integer percentage
# ticks up. n is the number of items completed so far. Call from the
# main process only.
function tick_progress(n, total, label, t_start)
    pct = (n * 100) ÷ total
    prev_pct = ((n - 1) * 100) ÷ total
    if pct > prev_pct
        elapsed = time() - t_start
        remaining = elapsed * (total - n) / n
        print(
            "\r  $(label): $(pct)% " *
            "($(fmt_time(elapsed)) elapsed, " *
            "$(fmt_time(remaining)) remaining)"
        )
    end
end

# tick_progress! is the Threads.@threads version. It atomically increments
# counter and delegates to tick_progress.
function tick_progress!(counter, total, label, t_start)
    n = Threads.atomic_add!(counter, 1) + 1
    tick_progress(n, total, label, t_start)
end

# scan_file reads only HDF5 dataset-size metadata (not pixel data) for one
# file and returns the projection keys whose images have width <= 2000.
# Workers call this during pass 1 so each worker has its own HDF5 context.
# The <= 2000 guard excludes merger galaxies, which tend to have very large
# images (e.g. galaxy 2 at 5550x5550 px).
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

# load_file reads all valid projections from one image file and returns
# the image data and associated metadata. Workers call this during pass 2
# so each worker has its own HDF5 context and no global lock contention.
function load_file(path, projs, fname, all_bands, Nbands, res)
    underscores = findall(isequal('_'), fname)
    obj_id = fname[1:underscores[2]-1]
    n_projs = length(projs)
    img_data = zeros(n_projs, Nbands, res, res)
    HDF5.h5open(path, "r") do file
        for (pi, proj) in enumerate(projs)
            # Ternary: load r, g, u bands for multi-band targets, or
            # only r for single-band targets.
            bands_list = all_bands ? ["r", "g", "u"] : ["r"]
            img = zeros(Nbands, res, res)
            for (bi, band) in enumerate(bands_list)
                band_data = read(file, proj * "/band_" * band)
                # h5py wrote (H, W); Julia reverses axes on read to (W, H).
                # Transpose back so spatial layout is correct.
                band_data = permutedims(band_data, (2, 1))
                if size(band_data) != (res, res)
                    band_data = Images.imresize(band_data, res, res)
                end
                img[bi, :, :] = parent(band_data)
            end
            img_data[pi, :, :, :] = img
        end
    end
    ids_chunk = fill(obj_id, n_projs)
    fnames_chunk = fill(fname, n_projs)
    return img_data, ids_chunk, Vector{String}(projs), fnames_chunk
end

function read_2d_shapes()
    function csv_read(path)
        # Header for the host file is
        #     galaxyID, FOV, pixel, view, band, b_a, PA, n, Re, Ie
        # Header for the satellite file is
        #     galaxyID, FOV, pixel, view, band, b_a, PA, n, Re, Ie,
        #     Mstar_ahf_cat, flag
        # Header for the octant file, from the current
        # gen_octant_shapes.py, is
        #     galaxyID, FOV, pixel, view, band, b_a, PA, n, Re, Ie_log10
        # plus a trailing `converged` column when the astrophot fitter
        # produced it. Both fitters write Ie_log10 (log10 of Ie); only
        # `converged` differs between them. Older octant files predating
        # the Ie_log10 rename instead have a plain `Ie` column, and so do
        # the host and satellite files above. In all of these cases the
        # column named `Ie` actually holds log10(Ie); the plain name just
        # predates the later rename. None of this affects Re, which is
        # what this function is here for, but it matters if this code
        # ever starts reading Ie too.
        dat = CSV.read(path, DataFrame, header=1)
        return dat
    end
    host_dat = csv_read(host_2d_shapes_path)
    sat_dat = csv_read(sat_2d_shapes_path)
    octant_dat = csv_read(octant_2d_shapes_path)

    # Keep only the columns common to all three sources: sat has extra
    # Mstar_ahf_cat/flag columns, and octant may have Ie_log10 + converged
    # instead of Ie, depending on which fitter produced it (see headers
    # above).
    common_cols = [
        :galaxyID, :FOV, :pixel, :view, :band, :b_a, :PA, :n, :Re
    ]
    host_dat = DataFrames.select(host_dat, common_cols)
    sat_dat = DataFrames.select(sat_dat, common_cols)
    octant_dat = DataFrames.select(octant_dat, common_cols)
    dat = vcat(host_dat, sat_dat, octant_dat)

    dat.galaxyID .= "object_" .* string.(dat.galaxyID)
    DataFrames.rename!(dat, :galaxyID => :Simulation)

    dat = dat[dat.band .== "band_r", :]

    # Drop rows with no usable radius (failed or not-yet-run fits write
    # blank/NaN Re), regardless of which of the three CSVs they came
    # from.
    has_re = map(re -> !ismissing(re) && !isnan(re), dat.Re)
    dat = dat[has_re, :]

    # A galaxy+view should appear at most once across the combined shape
    # CSVs; a duplicate means two different fits produced conflicting Re
    # values for the same projection, which is a real data problem.
    dupe_counts = DataFrames.combine(
        DataFrames.groupby(dat, [:Simulation, :view]),
        DataFrames.nrow => :n
    )
    @assert all(dupe_counts.n .== 1) (
        "Duplicate galaxy+view rows in the combined shape CSVs."
    )

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
        fname = "avg_sfrs_1.0Gyr_no_bound_filter.csv"
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
    println("Scanning host image directory..."); flush(stdout)
    host_fnames = filter(
        f -> isfile(joinpath(host_direc, f)) && endswith(f, ".hdf5"),
        readdir(host_direc)
    )
    host_paths = joinpath.(host_direc, host_fnames)
    println("Found $(length(host_fnames)) host files."); flush(stdout)
    println("Scanning satellite image directory..."); flush(stdout)
    sat_fnames = filter(
        f -> isfile(joinpath(sat_direc, f)) && endswith(f, ".hdf5"),
        readdir(sat_direc)
    )
    sat_paths = joinpath.(sat_direc, sat_fnames)
    println("Found $(length(sat_fnames)) satellite files."); flush(stdout)
    println("Scanning octant image directory..."); flush(stdout)
    octant_fnames = filter(
        f -> isfile(joinpath(octant_img_dir, f)) && endswith(f, ".hdf5"),
        readdir(octant_img_dir)
    )
    octant_paths = joinpath.(octant_img_dir, octant_fnames)
    println(
        "Found $(length(octant_fnames)) octant files."
    ); flush(stdout)
    files = [host_fnames; sat_fnames; octant_fnames]
    paths = [host_paths; sat_paths; octant_paths]

    baddies = [
        "object_1162",
        "object_280",
        # Gal 1238 is stripped but some vmap proj's have a tiny amount of gas
        # while others
        # are absent. The code doesn't drop the gal becasue some vmaps exist,
        # but then it errors because some are missing.
        "object_1238",
    ]
    is_bad = [any(occursin(baddy, f) for baddy in baddies) for f in files]

    println("Reading target dataframe..."); flush(stdout)
    if tgt_type == "3d"
        y_df = read_3d_tgt()
        all_bands = true
        Nbands = 3
    elseif tgt_type == "2d"
        y_df = read_2d_shapes()
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

    println(
        "Target dataframe loaded ($(nrow(y_df)) rows). " *
        "Building file mask..."
    ); flush(stdout)
    # For every file name, create an inner mask the size of y_df.Simulation
    # where a `true` marks the simulation in y_df (if any) whose name occurs
    # in that file name. If any row in that inner mask is true (although
    # there should be at most one), mark the file with a `true` in the
    # `in_tgt` outer mask.
    in_tgt = [
        any(occursin(obj * "_", f) for obj in y_df.Simulation)
        for f in files
    ]
    println(
        "File mask built: $(sum(in_tgt)) of $(length(files)) files " *
        "matched target dataframe."
    ); flush(stdout)

    good_files = files[.!is_bad .& in_tgt]
    good_paths = paths[.!is_bad .& in_tgt]
    println(
        "$(length(good_files)) good files after filtering baddies."
    ); flush(stdout)
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
    # pmap distributes files across separate worker processes so each
    # worker has its own HDF5 context and reads proceed without global
    # lock contention. The main process listens on prog1 and updates the
    # progress line as workers complete.
    println(
        "Pass 1: scanning $(Nfiles) files using $(nworkers()) workers..."
    ); flush(stdout)
    println("Creating RemoteChannel..."); flush(stdout)
    prog1 = RemoteChannel(() -> Channel{Nothing}(Nfiles))
    println("RemoteChannel created. Starting pmap..."); flush(stdout)
    t_p1 = time()
    p1_task = @async pmap(good_paths[1:Nfiles]) do path
        result = scan_file(path)
        put!(prog1, nothing)
        result
    end
    valid_projs_per_file = begin
        n = 0
        while n < Nfiles
            take!(prog1)
            n += 1
            tick_progress(n, Nfiles, "Pass 1", t_p1)
        end
        println() # Advance past the \r progress line.
        fetch(p1_task)
    end

    # Preflight check: verify that all orientations present in the image
    # data for each galaxy also exist in that galaxy's vmap file. Running
    # this after pass 1 but before the expensive pass 2 means a mismatch
    # aborts the job before we spend time loading pixel data.
    #
    # Build galaxy_id => Set{orientation} from pass 1 results.
    id_to_orients = Dict{String, Set{String}}()
    for fi in 1:Nfiles
        underscores = findall(isequal('_'), good_files[fi])
        obj_id = good_files[fi][1:underscores[2]-1]
        for proj in valid_projs_per_file[fi]
            orients = get!(id_to_orients, obj_id, Set{String}())
            push!(orients, proj)
        end
    end
    mismatches = Dict{String, Vector{String}}()
    mismatch_lock = ReentrantLock()
    ids_for_preflight = collect(keys(id_to_orients))
    n_preflight = length(ids_for_preflight)
    preflight_done = Threads.Atomic{Int}(0)
    preflight_start = time()
    println("Preflight: checking vmap orientation coverage...")
    Threads.@threads for id in ids_for_preflight
        id_int = parse(Int, replace(id, "object_" => ""))
        vmap_path = joinpath(maps_dir, "object_$(id_int)_vmap.hdf5")
        if isfile(vmap_path)
            # Read only top-level keys -- no pixel data loaded.
            vmap_keys = HDF5.h5open(vmap_path, "r") do f
                keys(f)
            end
            missing_orients = setdiff(id_to_orients[id], vmap_keys)
            if !isempty(missing_orients)
                lock(mismatch_lock) do
                    mismatches[id] = collect(missing_orients)
                end
            end
        end
        tick_progress!(
            preflight_done, n_preflight, "Preflight", preflight_start
        )
    end
    println() # Advance past the \r progress line.
    if !isempty(mismatches)
        mismatch_path = joinpath(output_dir, "vmap_mismatches.csv")
        open(mismatch_path, "w") do io
            println(io, "galaxy_id,missing_orientation")
            for (id, missing_orients) in sort(collect(mismatches))
                for orient in sort(missing_orients)
                    println(io, "$id,$orient")
                end
            end
        end
        println(
            "Preflight: $(length(mismatches)) galaxies have image " *
            "orientations missing from their vmaps. " *
            "Wrote details to $mismatch_path"
        )
        for (id, missing_orients) in sort(collect(mismatches))
            println("  $id: $missing_orients")
        end
        error("Orientation mismatch between images and vmaps.")
    end

    # Compute per-file starting row indices via prefix sum so that
    # the assembly step after pass 2 can place each file's rows without
    # any shared mutable state.
    counts = length.(valid_projs_per_file)
    offsets = cumsum([1; counts[1:end-1]])
    N_rows = sum(counts)

    X = zeros(N_rows, Nbands, res, res)
    ids_X = Vector{String}(undef, N_rows)
    fnames_sorted = Vector{String}(undef, N_rows)
    orientations = Vector{String}(undef, N_rows)

    # Pass 2: load and resize pixel data. Same pmap + RemoteChannel
    # pattern as pass 1. Each worker returns its image chunk and metadata;
    # the main process assembles them into X after all workers finish.
    println("Pass 2: loading images...")
    prog2 = RemoteChannel(() -> Channel{Nothing}(Nfiles))
    t_p2 = time()
    file_args = collect(zip(
        good_paths[1:Nfiles],
        valid_projs_per_file,
        good_files[1:Nfiles]
    ))
    p2_task = @async pmap(file_args) do args
        path, projs, fname = args
        result = load_file(path, projs, fname, all_bands, Nbands, res)
        put!(prog2, nothing)
        result
    end
    results = begin
        n = 0
        while n < Nfiles
            take!(prog2)
            n += 1
            tick_progress(n, Nfiles, "Pass 2", t_p2)
        end
        println() # Advance past the \r progress line.
        fetch(p2_task)
    end

    # Assemble worker results into X, ids_X, orientations, fnames_sorted.
    # Each file's rows are independent so @threads is safe here.
    Threads.@threads for fi in 1:Nfiles
        img_chunk, ids_chunk, orients_chunk, fnames_chunk = results[fi]
        base_row = offsets[fi]
        for pi in 1:counts[fi]
            row = base_row + pi - 1
            X[row, :, :, :] = img_chunk[pi, :, :, :]
            ids_X[row] = ids_chunk[pi]
            orientations[row] = orients_chunk[pi]
            fnames_sorted[row] = fnames_chunk[pi]
        end
    end
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
        # id_mask, orientation_mask, and mask are each
        # (length(ids_X), nrow(y_df)): rows are loaded images, columns
        # are target-dataframe rows, and an entry is true where that
        # image and that target row refer to the same galaxy
        # (id_mask), the same projection (orientation_mask), or both
        # (mask).
        id_mask = falses(length(ids_X), size(y_df)[1])
        orientation_mask = copy(id_mask)
        for i in 1:length(ids_X)
            id_mask[i, :] .= y_df[!, "Simulation"] .== ids_X[i]
            orientation_mask[i, :] .= y_df[!, "view"] .== orientations[i]
        end
        mask = id_mask .& orientation_mask
    elseif tgt_type in ("sfr", "avg_sfr")
        # mask is (length(ids_X), nrow(y_df)); see comment above.
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

    # Separate, independent lookup for Re. Unlike the target match above,
    # Re is an auxiliary input feature, not the training target, and a
    # missing radius is expected while Sersic fits are still in progress,
    # so we drop the image instead of crashing the run.
    shapes = read_2d_shapes()
    # re_mask is (length(ids_X), nrow(shapes)): rows are loaded images
    # (the target match above has already filtered these down), columns
    # are rows of `shapes`, true where the image and shapes row are the
    # same galaxy + projection.
    re_mask = falses(length(ids_X), nrow(shapes))
    for i in 1:length(ids_X)
        id_match = shapes[!, "Simulation"] .== ids_X[i]
        orient_match = shapes[!, "view"] .== orientations[i]
        re_mask[i, :] .= id_match .& orient_match
    end
    re_match_counts = vec(sum(re_mask, dims=2))
    # read_2d_shapes() already asserts there are no duplicate
    # Simulation+view rows in `shapes`, which guarantees every column
    # group above has at most one true entry per row. This assertion is
    # therefore redundant given that guarantee; it exists only to catch
    # a bug in the matching logic above should that guarantee ever stop
    # holding, not to catch new duplicate radius data.
    @assert all(re_match_counts .<= 1)
    re_keep = re_match_counts .== 1
    n_dropped = sum(.!re_keep)
    if n_dropped > 0
        println(
            "Dropping $(n_dropped) of $(length(re_keep)) images with " *
            "no matching radius in host_2d_shapes/sat_2d_shapes/" *
            "octant_shapes."
        ); flush(stdout)
    end
    re_indices = findfirst.(eachrow(re_mask))

    X = X[re_keep, :, :, :]
    ids_X = ids_X[re_keep]
    orientations = orientations[re_keep]
    fnames_sorted = fnames_sorted[re_keep]
    y_df = y_df[re_keep, :]
    # Re is an auxiliary input feature, not a training target, so we
    # keep it separate from y_df instead of merging it in.
    matched_shapes = shapes[re_indices[re_keep], :]
    Re_X = matched_shapes[:, "Re"]

    # Sanity-check the Re lookup as carefully as the old Python-side
    # get_radii() checked its CSV join: every kept image must line up
    # with its shapes row on both galaxy and orientation, and X, ids_X,
    # orientations, fnames_sorted, y_df, and Re_X must all come out the
    # same length.
    @assert all(ids_X .== matched_shapes[:, "Simulation"])
    @assert all(orientations .== matched_shapes[:, "view"])
    n = length(ids_X)
    @assert size(X, 1) == n
    @assert length(orientations) == n
    @assert length(fnames_sorted) == n
    @assert nrow(y_df) == n
    @assert length(Re_X) == n

    return ids_X, X, fnames_sorted, y_df, orientations, Re_X
end

function load_vmap(id, res)
    path = joinpath(maps_dir, "object_$(id)_vmap.hdf5")
    vmap = Dict{String, Dict}()
    if !isfile(path)
        return nothing
    end
    HDF5.h5open(path, "r") do file
        for orientation in keys(file)
            for data in keys(file[orientation])
                grp = read(file, orientation)
                # h5py wrote vmap as (H, W); Julia reverses axes to (W, H).
                # Transpose back so spatial layout is consistent with the
                # band images.
                grp["vmap"] = permutedims(grp["vmap"], (2, 1))
                vmap[orientation] = grp
            end
        end
    end
    return vmap
end

function build_training_data(tgt_type; Nfiles=nothing, save=false, res=256)
    println("build_training_data: calling load_images..."); flush(stdout)
    ids_X, X, files, y_df, orientations_X, Re_X = load_images(
        Nfiles=Nfiles,
        res=res,
        tgt_type=tgt_type
    )
    # Add in velocity map.
    VMAP = zeros(length(ids_X), 1, res, res)
    orientations = unique(orientations_X)
    ids_set = unique(ids_X)
    n_ids = length(ids_set)

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
            # ys as (features, N) means h5py reads (N, features). Re
            # follows the same (features, N) -> (N, features) pattern.
            X_out = permutedims(X, (4, 3, 2, 1))
            ys_out = permutedims(ys, (2, 1))
            Re_out = permutedims(reshape(Re_X, (:, 1)), (2, 1))
            for (label, data) in [
                        ["X", X_out],
                        ["obs_sorted", ids_X],
                        ["orientations", orientations_X],
                        ["file_names", files],
                        ["ys_sorted", ys_out],
                        ["Re", Re_out],
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

    return ids_X, y_df, ys, X, files, orientations_X, Re_X
end

end # module Dataset
