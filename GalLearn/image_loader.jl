module image_loader

    using HDF5
    using CSV
    using DataFrames
    using ProgressBars
    #import ImageFiltering
    import Images
    import StatsBase

    sat_direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/satellite/" *
        "ugrband_massmocks_final"
    host_direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/host/" *
        "ugrband_massmocks_final"
    gallearn_dir = "/export/nfs0home/pstaudt/projects/gal-learn/GalLearn"
    tgt_3d_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn/luke_protodata"
    tgt_2d_host_path = "/DFS-L/DATA/cosmo/kleinca/data/" *
        "AstroPhot_Host_Sersic-Copy1.csv"
    tgt_2d_sat_path = "/DFS-L/DATA/cosmo/kleinca/data/" *
        "AstroPhot_Sate_Sersic-Copy1.csv"
    output_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn"

    function process_file(
                fname,
                path,
                iX,
                X,
                shapeXimgs,
                obs_sorted,
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
                    bands = ["g"]
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
                    # skipping an image and `obs_sorted` will be shorter.
                    X = X[1 : end - 1, :, :, :]
                    shapeXimgs = size(X)[end - 1 : end]
                    # Go to the next projection without addint 1 to iX
                    continue
                end

                # Save this file's position. 
                underscores = findall(isequal('_'), fname)
                push!(obs_sorted, fname[1 : underscores[2] - 1])
                push!(orientations, proj)
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
            dat = CSV.read(
                tgt_path,
                DataFrame,
                header=[
                    "galaxyID",
                    "FOV",
                    "pixel",
                    "view",
                    "band",
                    "b_a_ave",
                    "PA",
                    "n",
                    "Re",
                    "Ie"
                ]
            )
            return dat
        end
        host_dat = csv_read(tgt_2d_host_path)
        sat_dat = csv_read(tgt_2d_sat_path)
        dat = vcat(host_dat, sat_dat)

        dat.galaxyID .= "object_" .* string.(dat.galaxyID)
        DataFrames.rename!(dat, :galaxyID => :Simulation)
        xydat = dat[dat.view .== "projection_xy", :]
        return xydat 
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
            ys = read_3d_tgt()
            all_bands = true
        elseif tgt_type == "2d"
            ys = read_2d_tgt()
            all_bands = false
        else
            throw(ArgumentError("`tgt_type` should be \"2d\" or \"3d\"."))
        end
        
        # For every file name, create a mask the size of ys.Simulation where a 
        # `true`
        # marks the row (if any) where `obj * "_"`
        # occurs in that file name. 
        # If any row in that mask is true (although there should be at most
        # one), the given file is marked with a `true`.
        in_tgt = [
            any(occursin(obj * "_", f) for obj in ys.Simulation) 
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

        if tgt_type == "2d"
            Nbands = 1
        elseif tgt_type == "3d"
            Nbands = 3
        else
            throw(ArgumentError("`tgt_type` should be \"2d\" or \"3d\"."))
        end
        N_proj = 3 # Number of projections
        global X = zeros(Nfiles * 3, Nbands, res, res) 
        shapeXimgs = size(X)[end - 1 : end]
        obs_sorted = String[]
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
                obs_sorted,
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
        return obs_sorted, X, fnames_sorted, ys
    end

    function load_data(tgt_type; Nfiles=nothing, save=false, res=256)
        obs_sorted, X, files, ys = load_images(
            Nfiles=Nfiles,
            res=res,
            tgt_type=tgt_type
        )

        indices = [
            findfirst(x -> x == val, ys.Simulation) for val in obs_sorted
        ]
        ys_sorted = ys[
            indices,
            :
        ]

        if tgt_type == "3d"
            ys_sorted = Array(ys_sorted[:, ["b/a", "c/a"]])
        elseif tgt_type == "2d"
            ys_sorted = Array(ys_sorted[:, "b_a_ave"])
            println("`ys` type: " * string(typeof(ys_sorted)))
            ys_sorted = reshape(ys_sorted, (size(ys_sorted)..., 1))
            println("`ys` shape: " * string(size(ys_sorted)))
        else
            throw(ArgumentError("`tgt_type` should be \"2d\" or \"3d\"."))
        end

        if save
            # Resolution
            fname = "gallearn_data_" * 
                string(res) * 
                "x" * 
                string(res) * 
                "_3proj"

            # Sample type
            if Nfiles !== nothing
                fname *= "_" * string(Nfiles) * "gal_subsample"
            end

            # 2d or 3d target data
            fname *= "_" * tgt_type * "_tgt"

            fname *= ".h5"

            h5open(joinpath(output_dir, fname), "w") do f
                for (label, data) in [
                            ["X", X],
                            ["obs_sorted", obs_sorted],
                            ["file_names", files],
                            ["ys_sorted", ys_sorted]
                        ]
                    println(
                        "Trying to save " 
                        * label 
                        * " of type $(typeof(data))"
                    )
                    write(f, label, data)
                end
            end
        end

        return obs_sorted, ys, ys_sorted, X, files
    end

end # module image_loader
