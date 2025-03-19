import Random
import HDF5
import PyCall
import Distributions

function generate_ellipse_mask(height, width, center, axes, angle)
    mask = falses(height, width)
    cx, cy = center
    a, b = axes
    cos_theta, sin_theta = cos(angle), sin(angle)
    
    for x in 1:width, y in 1:height
        dx, dy = x - cx, y - cy
        x_rot = dx * cos_theta + dy * sin_theta
        y_rot = -dx * sin_theta + dy * cos_theta
        if (x_rot^2 / a^2 + y_rot^2 / b^2) <= 1
            mask[y, x] = true
        end
    end
    return mask
end

function generate_ellipses(num_ellipses, img_size)
    height, width = img_size
    ellipses = zeros(Bool, num_ellipses, 1, height, width)
    ys = zeros(num_ellipses, 1)
    Random.seed!(42)  # For reproducibility
    
    for i in 1:num_ellipses
        cx = Random.rand(Distributions.Uniform(0.3, 0.7)) * width
        cy = Random.rand(Distributions.Uniform(0.3, 0.7)) * height
        a = Random.rand(Distributions.Uniform(0.1, 0.3)) * width
        b = Random.rand(Distributions.Uniform(0.1, 0.3)) * height
        angle = Random.rand(Distributions.Uniform(0, π))
        ellipses[i, 1, :, :] .= generate_ellipse_mask(
            height, width, (cx, cy), (a, b), angle)
	b, a = minmax(b, a)
	ys[i, 1] = b/a
    end
    return ellipses, ys
end

function save_ellipses_to_h5(filename, X, ys)
    push!(PyCall.pyimport("sys")."path", PyCall.pwd())
    paths = PyCall.pyimport("paths")
    data_path = paths.data
    HDF5.h5open(joinpath(data_path, filename), "w") do file
        HDF5.write(file, "X", X)
	HDF5.write(file, "ys_sorted", ys)
    end
end

# Parameters
num_ellipses = 2997
img_size = (256, 256)  # Adjust as needed

# Generate ellipses
X, ys = generate_ellipses(num_ellipses, img_size)

# Save to HDF5
save_ellipses_to_h5("ellipses.h5", X, ys)
