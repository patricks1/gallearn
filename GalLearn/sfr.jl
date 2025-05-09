using HDF5
import PyCall
import Plots
import Printf
import ProgressBars
import DataFrames
import CSV

ENV["GKSwstype"] = "100"
super_direc = "/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/"

function get_hosts()
    fname = super_direc * "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = read(file, "groupID")
       gal_ids = read(file, "galaxyID")
        return grp_ids, gal_ids
    end

    is_host = grp_ids .== -1
    host_ids = gal_ids[is_host]
    println(String("N hosts: $(length(host_ids))"))

    return host_ids
end

function get_both(; only_files=true)
    fname = super_direc * 
        "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    gal_ids = HDF5.h5open(fname) do file
        gal_ids = Int.(read(file, "galaxyID"))
        return gal_ids
    end

    println(String("N hosts and satellites: $(length(gal_ids))"))

    if only_files
        potential_files = [
            "particles_within_Rvir_object_" * 
                string(id) * 
                ".hdf5"
            for id in gal_ids
        ]
        direc = joinpath(super_direc, "objects_1200")
        println("Getting list of existing files.")
        #existing_files = filter(
        #    f -> isfile(joinpath(direc, f)) && endswith(f, ".hdf5"), 
        #    readdir(direc)
        #)
        existing_files = readdir(direc)
        println("Comparing to expected files.")
        exists = [p in existing_files for p in potential_files]
        gal_ids = gal_ids[exists]
    end

    return gal_ids
end

function get_sfrs(host_ids)
    sfrs_gals = Float64[]
    Mstar_gals = Float64[]
    for id in ProgressBars.ProgressBar(host_ids)
        id_str = string(id)
        fname = super_direc *
            "objects_1200/particles_within_Rvir_object_" * 
            #"objects_1200/bound_particle_filters_object_" * 
            id_str * 
            ".hdf5"
        if isfile(fname)
            h5open(fname, "r") do file
                global sfrs, gas_masses, Mstar, snap_time
                #println("\nAvailable keys:")
                #println(keys(file))
                sfrs = read(file, "gas_sfr")
                gas_masses = read(file, "gas_mass")
                Mstar = read(file, "Mstar")
                snap_time = read(file, "time")
            end
            sfr = sum(sfrs)
            #Printf.@printf("SFR: %.2f Msun / yr", sfr)
            push!(sfrs_gals, sfr) 
            push!(Mstar_gals, Mstar)
        end
    end

    Plots.histogram(
        log10.(sfrs_gals),
        #yscale=:log10,
        ylabel="N_gal",
        xlabel="log(SFR / [Msun / yr])",
        legend=false
    )
    Plots.savefig("hist.png") 

    Plots.scatter(
        log10.(Mstar_gals),
        log10.(sfrs_gals),
        ylabel="log(SFR / [Msun / yr])",
        xlabel="log(Mstar / Msun)",
        legend=false
    )
    Plots.savefig("scatter.png")

    return sfrs_gals
end

ids = get_both()
sfrs = get_sfrs(ids)
sfr_df = DataFrames.DataFrame(id=ids, sfr=sfrs) 
CSV.write("/DFS-L/DATA/cosmo/pstaudt/gallearn/sfrs.csv", sfr_df)
