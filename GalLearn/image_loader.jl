using HDF5
using CSV
using DataFrames
using ProgressBars
using ImageFiltering

direc = "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/satellite/ugrband_massmocks"
gallearn_dir = "/export/nfs0home/pstaudt/projects/gal-learn/GalLearn"
tgt_dir = "/export/nfs0home/lyxia/scripts/FIREBox/scripts/csvresults/" *
              "FIREBox_Allstars"

function read_tgt()
    files = readdir(tgt_dir)
    ys = CSV.read(joinpath(tgt_dir, "FIREBoxm9.csv"), DataFrame)
    for mclass in ["7", "8"] 
        ys_add = CSV.read(
            joinpath(tgt_dir, "FIREBoxm" * mclass * ".csv"), 
            DataFrame
        )
        ys = vcat(ys, ys_add)
    end
end

function load_images()
    files = filter(
        f -> isfile(joinpath(direc, f)) && endswith(f, ".hdf5"), 
        readdir(direc)
    )
    Nfiles = length(files)

    X = zeros(Nfiles, 3, 2, 2) 
    shapeXimgs = size(X)[end - 1 : end]
    objs_sorted = String[]

    open(joinpath(gallearn_dir, "image_loader_ram_use.txt"), "a") do f
        println(f, "Beginning.")
    end

    baddies = [
        "object_1162",
        "object_280",
    ]
    is_bad = [any(occursin(baddy, f) for baddy in baddies) for f in files]


    for (ifile, fname) in ProgressBar(enumerate(files[.!is_bad][1:10]))
        global X
        global shapeXimgs

        open(joinpath(gallearn_dir, "image_loader_ram_use.txt"), "a") do f
            println(f, fname)      
        end

        path = joinpath(direc, fname)
        h5open(path, "r") do file
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
            continue
        end

        # Save this file's position. 
        underscores = findall(isequal('_'), fname)
        println(fname)
        println(underscores)
        push!(objs_sorted, fname[1 : underscores[2] - 1])

        if shapeXimgs < shape_band
            pad = (shape_band .- shapeXimgs) ./ 2
            X = ImageFiltering.padarray(
                X, 
                Fill(0., (0, 0, Int(pad[1]), Int(pad[2])))
            )
        elseif shape_band < shapeXimgs
            pad = (shapeXimgs .- shape_band) ./ 2
            img = ImageFiltering.padarray(
                img,
                Fill(0., (0, Int(pad[1]), Int(pad[2])))
            )
        end           
            
        X[ifile, :, :, :] = img
        shapeXimgs = size(X)[end - 1 : end]
                    open(joinpath(
                                gallearn_dir, 
                                "image_loader_ram_use.txt"
                            ), "a") do f
                        println(f, shapeXimgs)
                    end
        GC.gc()
        open(joinpath(gallearn_dir, "image_loader_ram_use.txt"), "a") do f    
            println(f, "Memory used by X: $(Base.summarysize(X) / 1e9) GB")
        end
    println(objs_sorted)
    end
end

load_images()
