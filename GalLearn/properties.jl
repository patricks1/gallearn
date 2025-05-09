using HDF5
import PyCall
import Plots
import Printf

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

function test(host_ids)
    id = Int(host_ids[2])
    id_str = string(id)
    fname = direc *
        "objects_1200/particles_within_Rvir_object_" * 
        #"objects_1200/bound_particle_filters_object_" * 
        id_str * 
        ".hdf5"
    h5open(fname, "r") do file
        global sfrs, gas_masses, Mstar, snap_time
        println("\nAvailable keys:")
        println(keys(file))
        sfrs = read(file, "gas_sfr")
        gas_masses = read(file, "gas_mass")
        Mstar = read(file, "Mstar")
        snap_time = read(file, "time")
    end
    sfr = sum(sfrs)
    Printf.@printf("\nSFR: %.2f Msun / yr\n", sfr)
    Plots.histogram(gas_masses, yscale=:log10)
    Plots.savefig("hist.png") 

    fname = direc *
        "objects_760/particles_within_Rvir_object_" * 
        id_str * 
        ".hdf5"
    h5open(fname, "r") do file
        global last_sfrs, last_gas_masses, last_Mstar, last_snap_time
        last_sfrs = read(file, "gas_sfr")
        last_gas_masses = read(file, "gas_mass")
        last_Mstar = read(file, "Mstar")
        last_snap_time = read(file, "time")
    end
    Printf.@printf("time: %.1f Gyr\n", snap_time)
    Printf.@printf("last time: %.1f Gyr\n", last_snap_time)
    sfr_fr_m = (Mstar - last_Mstar) / (snap_time - last_snap_time) / 1.e9
    Printf.@printf("SFR from mass: %.2f Msun / yr\n", sfr_fr_m)
end

host_ids = get_hosts()
test(host_ids)
