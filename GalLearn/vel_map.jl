module vel_map

import HDF5

super_dir = "/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/"

function make_vel_map_gal(gal_id)
    id_str = string(gal_id)
    path = joinpath(
        super_dir,
        "objects_1200",
        "particles_within_Rvir_object_" * id_str * ".hdf5"
    )
    if isfile(path)
        host, stars, gas = HDF5.h5open(path, "r") do file
            println(keys(file))
            host = Dict()
            for feature in ("Xc", "Yc", "Zc", "VXc", "VYc", "VZc")
                host[feature] = read(file, feature)
            end
            stars = Dict()
            gas = Dict()
            for feature in (
                    "stellar_x",
                    "stellar_y",
                    "stellar_z",
                    "stellar_vx",
                    "stellar_vy",
                    "stellar_vz")
                stars[feature] = read(file, feature)
            for feature in (
                    "gas_x",
                    "gas_y",
                    "gas_z",
                    "gas_vx",
                    "gas_vy",
                    "gas_vz")
                gas[feature] = read(file, feature)
            end
            return host, stars, gas
        end
        host_pos = [host["Xc"], host["Yc"], host["Zc"]]
        stars_pos = hcat(
            stars["stellar_x"],
            stars["stellar_y"],
            stars["stellar_x"],
        )
        stars_v = hcat(
            stars["stellar_vx"],
            stars["stellar_vy"],
            stars["stellar_vz"]
        )
        gas_pos = hcat(
            stars["gas_x"],
            stars["gas_y"],
            stars["gas_x"],
        )
        gas_v = hcat(
            stars["gas_vx"],
            stars["gas_vy"],
            stars["gas_vz"]
        )
    end
    
    return host_pos, stars_pos, stars_v
end

end # module vel_map
