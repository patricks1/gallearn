import HDF5
import PyCall

direc = "/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/"

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

host_ids = get_hosts()
