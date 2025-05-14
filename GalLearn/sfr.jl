module sfr

using HDF5
import PyCall
import Plots
import Printf
import ProgressBars
import DataFrames
import CSV

ENV["GKSwstype"] = "100"
super_direc = "/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/"

function get_grp_id(gal_id)
    fname = super_direc * 
       "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = read(file, "groupID")
        gal_ids = read(file, "galaxyID")
        return grp_ids, gal_ids
    end
    grp_id = grp_ids[gal_id .== gal_ids]
    return grp_id
end

function get_sats()
    fname = super_direc * 
       "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = read(file, "groupID")
        gal_ids = Int.(read(file, "galaxyID"))
        return grp_ids, gal_ids
    end

    is_sat = grp_ids .!= -1
    sat_ids = gal_ids[is_sat]
    println(String("N satellites: $(length(sat_ids))"))

    return sat_ids
end

function get_hosts()
    fname = super_direc * 
       "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = read(file, "groupID")
        gal_ids = Int.(read(file, "galaxyID"))
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
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = Int.(read(file, "groupID"))
        gal_ids = Int.(read(file, "galaxyID"))
        return grp_ids, gal_ids
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

    return gal_ids, grp_ids
end

function get_bound_particles(id)
    path = joinpath(
        super_direc,
        "objects_1200",
        "bound_particle_filters_object_" * string(id) * ".hdf5"
    )
    particle_ids = HDF5.h5open(path, "r") do file
        particle_ids = Int.(read(file, "particleIDs"))
        return particle_ids
    end
    return particle_ids
end

function get_sfrs(ids, grp_ids)
    sfrs_gals = Float64[]
    Mstar_gals = Float64[]
    for (gal_id, grp_id) in ProgressBars.ProgressBar(zip(ids, grp_ids))
        id_str = string(gal_id)
        fname = super_direc *
            "objects_1200/particles_within_Rvir_object_" * 
            id_str * 
            ".hdf5"
        if isfile(fname)
            sfrs, gas_masses, Mstar, snap_time, gas_ids = h5open(
                        fname, 
                        "r"
                    ) do file
                sfrs = read(file, "gas_sfr")
                gas_masses = read(file, "gas_mass")
                gas_ids = Int.(read(file, "gas_id"))
                Mstar = read(file, "Mstar")
                snap_time = read(file, "time")
                return sfrs, gas_masses, Mstar, snap_time, gas_ids
            end
            if grp_id != -1
                # If the galaxy is not a host, filter for only bound particles.
                bound_ids = get_bound_particles(gal_id)
                # Narrow down the bound gas IDs so the `in.` below is faster.
                bound_gas_ids = intersect(gas_ids, bound_ids)
                is_bound = in.(gas_ids, Ref(Set(bound_gas_ids)))
                #Printf.@printf("\n%.0f bound gas particles\n\n", sum(is_bound))
                @assert sum(is_bound) != 0 
                sfrs = sfrs[is_bound]
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

function get_all_sfrs(;save=false)
    grp_ids, gal_ids = get_both()
    sfrs = get_sfrs(gal_ids, grp_ids)
    sfr_df = DataFrames.DataFrame(id=ids, sfr=sfrs) 
    if save
        CSV.write("/DFS-L/DATA/cosmo/pstaudt/gallearn/sfrs.csv", sfr_df)
    end
    return sfr_df
end

end # module sfr
