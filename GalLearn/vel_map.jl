module vel_map

import HDF5

super_dir = "/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/"

function make_vel_map_gal(id)
    id_str = string(gal_id)
    path = joinpath(
        super_dir,
        "objects_1200",
        "particles_within_Rvir_object_" * id_str * ".hdf5"
    )
    if isfile(path)
        HDF5.h5open(path, "r") do file
            println(keys(file))
        end
    end
end

end # module vel_map
