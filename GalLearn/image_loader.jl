module image_loader

using HDF5
using CSV
using DataFrames
using ProgressBars
#import ImageFiltering
import Images
import StatsBase

#sat_direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/satellite/" *
#    "ugrband_massmocks_final"
sat_direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/satellite/" *
    "band_ugr"
#host_direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/host/" *
#    "ugrband_massmocks_final"
host_direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/host/" *
    "band_ugr"
gallearn_dir = "/export/nfs0home/pstaudt/projects/gal-learn/GalLearn"
tgt_3d_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn/luke_protodata"
tgt_sfr_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn/"
tgt_2d_host_path = "/DFS-L/DATA/cosmo/kleinca/data/" *
    "AstroPhot_NewHost_bandr_Rerun_Sersic.csv"
# Need to use AllMeasure satellite file because the other file only has
# g-band data, and we need r-band.
tgt_2d_sat_path = "/DFS-L/DATA/cosmo/kleinca/data/" *
    "DataWithMockImagesWithBadExtinction/" *
    "AstroPhot_Sate_Sersic_AllMeasure.csv"
output_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn"

function process_file(
            fname,
            path,
            iX,
            X,
            shapeXimgs,
            ids_X,
            fnames_sorted,
            gallearn_dir,
            all_bands,
            orientations
        )
    open(joinpath(gallearn_dir, "image_loader_ram_use.txt"), "a") do f
        println(f, fname)      
    end

    if all_bands
        Nbands = 3
    else
        Nbands = 1
    end

    h5open(path, "r") do file
        #println("reading $fname")
        global shape_band
        for proj in ["projection_xy/", "projection_yz/", "projection_zx/"]
            shape_band = nothing
            img = nothing
            if all_bands
                bands = ["g", "u", "r"]
            else
                bands = ["r"]
            end
            for (i, band) in enumerate(bands)
                band = "band_" * band
                band_img = read(file, proj * band)
                shape_band = size(band_img) 
                if shape_band[end] > 2000
                    # Leave img as nothing if the image is too big. Once we 
                    # break
                    # this loop here, we'll continue to the next file 
                    # below.
                    break 
                end
                if i == 1 
                    img = zeros(Nbands, shape_band...) 
                elseif size(band_img) != shape_band
                    error("Bands have different shapes.")
                end
                img[i, :, :] = band_img 
            end

            if shape_band[end] > 2000
                # Image is too big; we should move to the next file. If 
                # this is the
                # case, img == nothing at this point, given the break 
                # above.
                open(joinpath(
                        gallearn_dir, "image_loader_ram_use.txt"), 
                        "a") do f
                    println(f, "Skipping " * fname)
                end
                # Make X one row smaller than we were expecting, since 
                # we're
                # skipping an image and `ids_X` will be shorter.
                X = X[1 : end - 1, :, :, :]
                shapeXimgs = size(X)[end - 1 : end]
                # Go to the next projection without addint 1 to iX
                continue
            end

            # Save this file's position. 
            underscores = findall(isequal('_'), fname)
            push!(ids_X, fname[1 : underscores[2] - 1])
            push!(orientations, replace(proj, "/" => ""))
            push!(fnames_sorted, fname)

            #if shapeXimgs < shape_band
            #    #pad = (shape_band .- shapeXimgs) ./ 2
            #    #X = ImageFiltering.padarray(
            #    #    X, 
            #    #    Fill(0., (0, 0, Int(pad[1]), Int(pad[2])))
            #    #)
            #    X = Images.imresize(X, size(X)[1:2]..., shape_band...)
            #elseif shape_band < shapeXimgs
            #    #pad = (shapeXimgs .- shape_band) ./ 2
            #    #img = ImageFiltering.padarray(
            #    #    img,
            #    #    Fill(0., (0, Int(pad[1]), Int(pad[2])))
            #    #)
            #    img = Images.imresize(img, Nbands, shapeXimgs...)
            #end           

            if shapeXimgs != shape_band
                if Nbands > 1
                    img = Images.imresize(
                        img,
                        Nbands,
                        shapeXimgs...
                    )
                else
                    innerimg = img[1, :, :]
                    img = Images.imresize(innerimg, shapeXimgs...)
                    img = reshape(img, (1, size(img)...))
                end
            end
                
            X = parent(X) # Removing ridiculous OffsetArray indexing
            X[iX, :, :, :] = img
            shapeXimgs = size(X)[end - 1 : end]
            open(joinpath(
                        gallearn_dir, 
                        "image_loader_ram_use.txt"
                    ), "a") do f
                println(f, shapeXimgs)
                println(
                    f, 
                    "Memory used by X: $(Base.summarysize(X) / 1e9) GB"
                )
            end

            # Set the index for the next element of X to create. Note that 
            # this
            # advancement only happens if we don't skip the image we 
            # evaluate.
            # For instances where we skip, there's a continue statement 
            # above
            # that runs *before* we add to iX.
            iX += 1
        end
    end
    
    return X, shapeXimgs, iX
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
            tgt_type="3d"
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
    files = [host_fnames; sat_fnames]
    paths = [host_paths; sat_paths]

    open(joinpath(gallearn_dir, "image_loader_ram_use.txt"), "a") do f
        println(f, "Beginning.")
    end

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
        all_bands= true
        Nbands = 3
    else
        throw(ArgumentError(
            "`tgt_type` should be \"2d\", \"3d\", \"sfr\", or \"avg_sfr\"."
        ))
    end
    
    # For every file name, create an inner mask the size of y_df.Simulation 
    # where 
    # a 
    # `true`
    # marks the simulation in y_df (if any) whose name 
    # occurs in that file name. 
    # If any row in that inner mask is true (although there should be at most
    # one), the given file is marked with a `true` in the `in_tgt` outer mask.
    in_tgt = [
        any(occursin(obj * "_", f) for obj in y_df.Simulation) 
        for f in files
    ]

    good_files = files[.!is_bad .& in_tgt]
    good_paths = paths[.!is_bad .& in_tgt]
    if Nfiles === nothing
        # If the user hasn't specified the number of files to run through, 
        # then
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

    N_proj = 3 # Number of projections
    global X = zeros(Nfiles * 3, Nbands, res, res) 
    shapeXimgs = size(X)[end - 1 : end]
    ids_X = String[]
    fnames_sorted = String[]
    orientations = String[]

    iX = 1
    for item in ProgressBar(
                zip(good_files[1:Nfiles], good_paths[1:Nfiles])
            )
        fname, path = item
        X, shapeXimgs, iX = process_file(
            fname,
            path,
            iX,
            X,
            shapeXimgs,
            ids_X,
            fnames_sorted,
            gallearn_dir,
            all_bands,
            orientations
        )
    end
    # Get rid of the ridiculous OffsetArray indexing
    X = parent(X)
    println("X shape: $(size(X))")
    if logandscale
        # Turn zeros into the smallest value greater than zero to avoid 
        # -Inf in
        # the logs.
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
            # For every id_X, generate a vector of booleans, corresponding to
            # the rows in y_df where the Simulation matches id_X. (There should
            # only be one match for each id_X.)
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

    return ids_X, X, fnames_sorted, y_df, orientations, mask
end

function load_data(tgt_type; Nfiles=nothing, save=false, res=256)
    ids_X, X, files, y_df, orientations_X = load_images(
        Nfiles=Nfiles,
        res=res,
        tgt_type=tgt_type
    )

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
            "_3proj_wsat"

        # Sample type
        if Nfiles !== nothing
            fname *= "_" * string(Nfiles) * "gal_subsample"
        end

        # 2d, 3d, sfr, or avg_sfr target data
        fname *= "_" * tgt_type * "_tgt"

        fname *= ".h5"

        h5open(joinpath(output_dir, fname), "w") do f
            for (label, data) in [
                        ["X", X],
                        ["obs_sorted", ids_X],
                        ["orientations", orientations_X],
                        ["file_names", files],
                        ["ys_sorted", ys],
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

end # module image_loader
