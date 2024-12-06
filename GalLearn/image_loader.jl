module image_loader

    using HDF5
    using CSV
    using DataFrames
    using ProgressBars
    #import ImageFiltering
    import Images

    direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/satellite/" *
        "ugrband_massmocks_final"
    direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/host/" *
        "ugrband_massmocks_final"
    gallearn_dir = "/export/nfs0home/pstaudt/projects/gal-learn/GalLearn"
    # tgt_dir = "/export/nfs0home/lyxia/scripts/FIREBox/scripts/csvresults/" *
    #     "FIREBox_Allstars"
    tgt_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn/luke_protodata"
    output_dir = "/DFS-L/DATA/cosmo/pstaudt/gallearn"

    function process_file(
                fname,
                iX,
                X,
                shapeXimgs,
                obs_sorted,
                fnames_sorted,
                direc,
                gallearn_dir
            )
        open(joinpath(gallearn_dir, "image_loader_ram_use.txt"), "a") do f
            println(f, fname)      
        end

        path = joinpath(direc, fname)
        h5open(path, "r") do file
            #println("reading $fname")
            global shape_band
            shape_band = nothing
            img = nothing
            for (i, band) in enumerate(["g", "u", "r"])
                band = "band_" * band
                band_img = read(file, "projection_xy/" * band)
                shape_band = size(band_img) 
                if shape_band[end] > 2000
                    # Leave img as nothing if the image is too big. Once we 
                    # break
                    # this loop here, we'll continue to the next file below.
                    break 
                end
                if i == 1 
                    img = zeros(3, shape_band...) 
                elseif size(band_img) != shape_band
                    error("Bands have different shapes.")
                end
                img[i, :, :] = band_img 
            end
        end
        
        if shape_band[end] > 2000
            # Image is too big; we should move to the next file. If this is the
            # case, img == nothing at this point, given the break above.
            open(joinpath(gallearn_dir, "image_loader_ram_use.txt"), "a") do f
                println(f, "Skipping " * fname)
            end
            # Make X one row smaller than we were expecting, since we're
            # skipping an image and `obs_sorted` will be shorter.
            X = X[1 : end - 1, :, :, :]
            shapeXimgs = size(X)[end - 1 : end]
            # Exit without addint 1 to iX
            return X, shapeXimgs, iX
        end

        # Save this file's position. 
        underscores = findall(isequal('_'), fname)
        push!(obs_sorted, fname[1 : underscores[2] - 1])
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
        #    img = Images.imresize(img, 3, shapeXimgs...)
        #end           
        if shapeXimgs != shape_band
            img = Images.imresize(img, 3, shapeXimgs...)
        end
            
        X = parent(X) # Removing ridiculous OffsetArray indexing
        X[iX, :, :, :] = img
        shapeXimgs = size(X)[end - 1 : end]
        open(joinpath(
                    gallearn_dir, 
                    "image_loader_ram_use.txt"
                ), "a") do f
            println(f, shapeXimgs)
            println(f, "Memory used by X: $(Base.summarysize(X) / 1e9) GB")
        end

        # Set the index for the next element of X to set. Note that this
        # advancement only happens if we don't skip the image we evaluate.
        # For instances where we skip, there's another return statement above
        # that runs *before* we add to iX.
        iX += 1

        return X, shapeXimgs, iX
    end

    function read_tgt()
        files = readdir(tgt_dir)
        ys = CSV.read(joinpath(tgt_dir, "FIREBoxm9.csv"), DataFrame)
        for mclass in ["7", "8", "10"] 
            ys_add = CSV.read(
                joinpath(tgt_dir, "FIREBoxm" * mclass * ".csv"), 
                DataFrame
            )
            ys = vcat(ys, ys_add)
        end
        return ys
    end

    function load_images(; Nfiles=nothing, logandscale=false, res=500)
        files = filter(
            f -> isfile(joinpath(direc, f)) && endswith(f, ".hdf5"), 
            readdir(direc)
        )

        open(joinpath(gallearn_dir, "image_loader_ram_use.txt"), "a") do f
            println(f, "Beginning.")
        end

        baddies = [
            "object_1162",
            "object_280",
        ]
        is_bad = [any(occursin(baddy, f) for baddy in baddies) for f in files]

        ys = read_tgt()
        in_tgt = [
            any(occursin(obj * "_", f) for obj in ys.Simulation) 
            for f in files
        ]

        good_files = files[.!is_bad .& in_tgt]
        if Nfiles === nothing
            # If the user hasn't specified the number of files to run through, 
            # then
            # run through all of them.
            Nfiles = length(good_files)
        end

        global X = zeros(Nfiles, 3, res, res) 
        shapeXimgs = size(X)[end - 1 : end]
        obs_sorted = String[]
        fnames_sorted = String[]

        iX = 1
        for fname in ProgressBar(
                    good_files[1:Nfiles]
                )
            X, shapeXimgs, iX = process_file(
                fname,
                iX,
                X,
                shapeXimgs,
                obs_sorted,
                fnames_sorted,
                direc,
                gallearn_dir
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
        return obs_sorted, X, fnames_sorted
    end

    function load_data(; Nfiles=nothing, save=false, res=500)
        obs_sorted, X, files = load_images(Nfiles=Nfiles, res=res)
        ys = read_tgt() 
        ys_sorted = ys[[
                findfirst(x -> x == val, ys.Simulation) for val in obs_sorted
            ], :]
        ys_sorted = Array(ys_sorted[:, ["b/a", "c/a"]])

        #println(size(ys))
        #println(size(ys_sorted))
        #println(size(obs_sorted))
        #println(size(X))

        if save
            fname = "gallearn_data_" * string(res) * "x" * string(res)
            if Nfiles !== nothing
                fname *= "_subsample"
            end
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
