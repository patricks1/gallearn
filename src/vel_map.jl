module vel_map

import HDF5
import PyCall
import Plots
import Printf

ENV["GKSwstype"] = "100"

super_dir = "/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/"

function make_vel_map_gal(gal_id)
    uci = PyCall.pyimport("UCI_tools")
    cosmo = PyCall.pyimport("astropy.cosmology")
    astropy = PyCall.pyimport("astropy")

    id_str = string(gal_id)
    path = joinpath(
        super_dir,
        "objects_1200",
        "particles_within_Rvir_object_" * id_str * ".hdf5"
    )
    if isfile(path)
        data = Dict()
        host, stars, gas = HDF5.h5open(path, "r") do file
            println("Datasets in " * path * ":") 
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
                    "stellar_vz",
                    # Scale factor at each star's birth
                    "stellar_tform")
                stars[feature] = read(file, feature)
            end
            for feature in (
                    "gas_x",
                    "gas_y",
                    "gas_z",
                    "gas_vx",
                    "gas_vy",
                    "gas_vz",
                    "gas_metal_01",
                    "gas_u",
                    "gas_mass", # Note, this is units of 1e10 M_sun
                    "gas_ne")
                gas[feature] = read(file, feature)
            end
            return host, stars, gas
        end

        host_pos = [host["Xc"], host["Yc"], host["Zc"]]

        stars_pos = hcat(
            stars["stellar_x"],
            stars["stellar_y"],
            stars["stellar_z"]
        )
        stars_v = hcat(
            stars["stellar_vx"],
            stars["stellar_vy"],
            stars["stellar_vz"]
        )

        gas_pos = hcat(
            gas["gas_x"],
            gas["gas_y"],
            gas["gas_z"]
        )
        gas_v = hcat(
            gas["gas_vx"],
            gas["gas_vy"],
            gas["gas_vz"]
        )
        gas_temps = uci.tools.calc_temps(
            gas["gas_metal_01"],
            gas["gas_ne"],
            gas["gas_u"]
        )
        data = Dict([
            ("pos_gas", gas_pos),
            ("vel_gas", gas_v),
            ("temp", gas_temps),
            # Convert mass to M_sun from 1e10 M_sun
            ("mass_gas", gas["gas_mass"] * 1.e10),
            ("pos_star", stars_pos),
            ("vel_star", stars_v),
            ("a_form_star", stars["stellar_tform"])
        ])
        #Plots.histogram(gas["gas_mass"])
        #Plots.savefig("hist.png")

        stars_rs = sqrt.(sum(abs2, stars_pos; dims=2)) |> vec
        stars_in_fov = stars_rs .<= 34.
        stars_pos = stars_pos[stars_in_fov, :]
        stars_v = stars_v[stars_in_fov, :]

        gas_rs = sqrt.(sum(abs2, gas_pos; dims=2)) |> vec
        gas_in_fov = gas_rs .<= 34.
        gas_pos = gas_pos[gas_in_fov, :]
        gas_v = gas_v[gas_in_fov, :]
                             
        println("Making the velocity map.")
        maps = uci.vel_map.plot(
            stars_pos[:, [2, 1, 3]],
            stars_v[:, [2, 1, 3]],
            gas_pos[:, [2, 1, 3]],
            gas_v[:, [2, 1, 3]],
            "obj_1",
            "600",
            1,
            1,
            save_plot=true 
        )

    end
    
    return host_pos, stars_pos, stars_v, data 
end

end # module vel_map
