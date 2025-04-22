using HDF5
import PyCall
import Plots
import Printf
import ProgressBars

direc = "/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/"

function get_hosts()
    fname = direc * "global_sample_data/global_sample_data_snapshot_1200.hdf5"
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


function get_sfrs(host_ids)
    sfr_gals = Float64[]
    Mstar_gals = Float64[]
    for id in ProgressBars.ProgressBar(host_ids)
        id = Int(id)
        id_str = string(id)
        fname = direc *
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
            push!(sfr_gals, sfr) 
            push!(Mstar_gals, Mstar)
        end
    end
    Plots.histogram(
        log10.(sfr_gals),
        #yscale=:log10,
        ylabel="N_gal",
        xlabel="log(SFR / [Msun / yr])",
        legend=false
    )
    Plots.savefig("hist.png") 

    Plots.scatter(
        log10.(Mstar_gals),
        log10.(sfr_gals),
        ylabel="log(SFR / [Msun / yr])",
        xlabel="log(Mstar / Msun)",
    )
    Plots.savefig("scatter.png")
end


host_ids = get_hosts()
get_sfrs(host_ids)
