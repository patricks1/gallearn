module firebox_io 

using HDF5
import PyCall
import Plots
import Printf
import ProgressBars
import DataFrames
import CSV
import Statistics
import IndexedDFs

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
        grp_ids = Int.(read(file, "groupID"))
        gal_ids = Int.(read(file, "galaxyID"))
        return grp_ids, gal_ids
    end

    is_sat = grp_ids .!= -1
    sat_ids = gal_ids[is_sat]
    grp_ids = grp_ids[is_sat]
    println(String("N satellites: $(length(sat_ids))"))

    return sat_ids, grp_ids
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
        grp_ids = grp_ids[exists]
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

function summarize_gals(;save=false)
    ids, grp_ids = get_both()
    summary_fields = String[]

    df = DataFrames.DataFrame(
        id=ids,
        grp_id=grp_ids,
    )
    idf = IndexedDFs.IndexedDF(df, "id")


    for (i, (gal_id, grp_id)) in ProgressBars.ProgressBar(
                enumerate(zip(ids, grp_ids))
            )
        id_str = string(gal_id)
        fname = super_direc *
            "objects_1200/particles_within_Rvir_object_" * 
            id_str * 
            ".hdf5"
        if isfile(fname)
            h5open(
                        fname, 
                        "r"
                    ) do file
                # Use the first file to determine which data fields summarize
                # the galaxy as a whole as opposed to being an array of
                # particle values.
                if i == 1
                    all_fields = keys(file)
                    for key in all_fields
                        if read(file, key) isa Number
                            push!(summary_fields, key)
                            # Make the col for the df:
                            idf[:, key] = Any[fill(nothing, length(ids))...] 
                            # Add this gal's value to the col:
                            idf[gal_id, key] = read(file, key)      
                        end
                    end
                else
                    for key in summary_fields
                        idf[gal_id, key] = read(file, key)      
                    end
                end
            end
        else
            if verbose
                println("Could not find file " * fname)
            end
            # Drop the galaxy
            deleteat!(idf, gal_id)
        end
    end
    
    if save
        CSV.write(
            "/DFS-L/DATA/cosmo/pstaudt/gallearn/firebox_summary_stats.csv",
            idf.df 
        )
    end
    
    return idf
end

function get_sfrs(
            ids,
            grp_ids;
            make_plots=true,
            verbose=false
        )
    df = DataFrames.DataFrame(
        id=ids,
        grp_id=grp_ids,
        sfr=Any[fill(nothing, length(ids))...],
        sfr_unfiltered=Any[fill(missing, length(ids))...],
        ssfr=Any[fill(nothing, length(ids))...],
        Mstar=Any[fill(nothing, length(ids))...],
        bound_frac=Float64[fill(1., length(ids))...]
    )
    idf = IndexedDFs.IndexedDF(df, "id")

    sfrs_gals = Float64[]
    missing_files = Int64[]
    zero_bound = Int64[]

    for (gal_id, grp_id) in ProgressBars.ProgressBar(zip(ids, grp_ids))
        id_str = string(gal_id)
        if verbose
            println(
                "\nObject " * id_str * " has group ID " * string(grp_id) * "."
            )
        end
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
            if Mstar == 0.
                deleteat!(idf, gal_id)
                continue # Skip this galaxy.
            end
            if grp_id != -1
                # If the galaxy is not a host, filter for only bound particles.
                bound_ids = get_bound_particles(gal_id)
                # Narrow down the bound gas IDs so the `in.` below is faster.
                bound_gas_ids = intersect(gas_ids, bound_ids)
                is_bound = in.(gas_ids, Ref(Set(bound_gas_ids)))
                bound_frac = Statistics.mean(is_bound)
                if verbose
                    Printf.@printf(
                        "%.0f%% of gas particles in the object are bound.",
                        bound_frac * 100.
                    )
                    println("\nFirst 100 bound particle IDs:")
                    println(bound_ids[1:100])
                end
                idf[gal_id, "bound_frac"] = bound_frac
                if sum(is_bound) == 0 
                    # If there's no overlap between the particle IDs in the
                    # bound particles file and those in the Rvir file, there's
                    # probably a problem. We should drop those galaxies for
                    # now.
                    deleteat!(idf, gal_id)
                    push!(zero_bound, gal_id)
                    continue # Skip to the next gal.
                end
                idf[gal_id, "sfr_unfiltered"] = sum(sfrs)
                sfrs = sfrs[is_bound]
            end
            sfr = sum(sfrs)
            #Printf.@printf("SFR: %.2f Msun / yr", sfr)
            idf[gal_id] = (sfr=sfr, ssfr=sfr/Mstar, Mstar=Mstar)
            push!(sfrs_gals, sfr) 
        else
            if verbose
                println("Could not find file " * fname)
            end
            push!(missing_files, gal_id)
            # Drop the galaxy
            deleteat!(idf, gal_id)
        end
    end

    if make_plots
        Plots.histogram(
            idf.ssfr,
            #yscale=:log10,
            ylabel="N_gal",
            xlabel="SFR / M_stellar [yr^-1])",
            legend=false
        )
        Plots.savefig("hist.png") 

        Plots.scatter(
            log10.(idf.Mstar),
            idf.ssfr,
            ylabel="SFR / M_stellar [yr^-1])",
            xlabel="log(Mstar / Msun)",
            legend=false
        )
        Plots.savefig("scatter.png")
    end

    println("\n$(length(missing_files)) missing satellite files:") 
    println(missing_files)
    println(
        "\n$(length(zero_bound)) satellites with no overlap with" *
        " bound_particle file:"
    )
    println(zero_bound)
    
    return idf
end

function get_avg_sfrs(ids, grp_ids)
    uci = PyCall.pyimport("UCI_tools")    
    cosmo = PyCall.pyimport("astropy.cosmology")
    astropy = PyCall.pyimport("astropy")

    z_1Gyr = cosmo.z_at_value(
        cosmo.Planck13.lookback_time, 
        1. * astropy.units.Gyr
    )[1]
    a_1Gyr = 1. / (z_1Gyr + 1.)

    df = DataFrames.DataFrame(
        id=ids,
        grp_id=grp_ids,
        sfr=Any[fill(nothing, length(ids))...],
        ssfr_unfiltered=Any[fill(missing, length(ids))...],
        ssfr=Any[fill(nothing, length(ids))...],
        Mstar=Any[fill(nothing, length(ids))...],
        bound_star_frac=Float64[fill(1., length(ids))...],
    )
    idf = IndexedDFs.IndexedDF(df, "id")

    for (gal_id, grp_id) in ProgressBars.ProgressBar(zip(ids, grp_ids))
        id_str = string(gal_id)
        path = joinpath(
            super_direc,
            "objects_1200",
            "particles_within_Rvir_object_" * id_str * ".hdf5"
        )
        if isfile(path)
            file_contents = h5open(path, "r") do file
                if !("stellar_tform" in keys(file))
                    # If there's no scale factor of stellar formation
                    # information for this galaxy, skip it and delete it from
                    # the DataFrame.
                    return -1
                end
                stellar_scale_facs = read(file, "stellar_tform")
                Mstar = read(file, "Mstar")
                # Stellar masses in units of M_s
                stellar_masses = read(file, "stellar_mass") * 1.e10
                stellar_ids = read(file, "stellar_id")
                return stellar_scale_facs, Mstar, stellar_masses, stellar_ids
            end
            if file_contents == -1
                # If there's no scale factor of stellar formation
                # information for this galaxy, skip it and delete it from
                # the DataFrame.
                deleteat!(idf, gal_id)
                continue
                
            end
            (
                stellar_scale_facs,
                Mstar,
                stellar_masses,
                stellar_ids 
            ) = file_contents

            is_younger_1Gyr = stellar_scale_facs .>= a_1Gyr
            masses_younger_1Gyr = stellar_masses[is_younger_1Gyr]
            # SFR in M_sun / yr:

            if grp_id != -1
                # If the galaxy is not a host, filter for only bound particles.
                bound_ids = get_bound_particles(gal_id)
                # Narrow down the bound IDs so the `in.` below is faster.
                bound_stellar_ids = intersect(stellar_ids, bound_ids)
                is_bound = in.(stellar_ids, Ref(Set(bound_stellar_ids)))
                bound_frac = Statistics.mean(is_bound)
                idf[gal_id, "bound_star_frac"] = bound_frac
                if sum(is_bound) == 0 
                    # If there's no overlap between the particle IDs in the
                    # bound particles file and those in the Rvir file, there's
                    # probably a problem. We should drop those galaxies for
                    # now.
                    deleteat!(idf, gal_id)
                    continue # Skip to the next gal.
                end
                idf[gal_id, "ssfr_unfiltered"] = (
                    sum(masses_younger_1Gyr) / 1.e9 / Mstar
                )
                masses_younger_1Gyr = stellar_masses[
                    is_bound .& is_younger_1Gyr
                ]
            end

            # SFR in units of M_solar / yr
            sfr = sum(masses_younger_1Gyr) / 1.e9
            Mstar_fr_particles = sum(stellar_masses)

            idf[gal_id, "Mstar"] = Mstar
            idf[gal_id, "sfr"] = sfr
            idf[gal_id, "ssfr"] = sfr / Mstar

        end
    end

    return idf
end

function compare_sats_b4_filtering()
    gal_ids, grp_ids = get_sats()
    sfr_df = get_sfrs(
        gal_ids,
        grp_ids,
        make_plots=false,
    )
    return sfr_df
end

function get_all_avg_sfrs(;save=false)
    gal_ids, grp_ids = get_both()
    sfr_df = get_avg_sfrs(gal_ids, grp_ids)
    if save
        CSV.write("/DFS-L/DATA/cosmo/pstaudt/gallearn/avg_sfrs.csv", sfr_df.df)
    end
    return sfr_df
end

function get_all_sfrs(;save=false)
    gal_ids, grp_ids = get_both()
    sfr_df = get_sfrs(gal_ids, grp_ids, make_plots=true)
    if save
        CSV.write("/DFS-L/DATA/cosmo/pstaudt/gallearn/sfrs.csv", sfr_df.df)
    end
    return sfr_df
end

end # module firebox_io 
