module GalLearnConfig

import ConfParser
import Pkg

export read_config

# Walk up the directory tree from the active project until finding a
# Project.toml that has a `name` field, then return ~/config_<name>.ini.
# This handles sub-projects (e.g. scripts/) that intentionally omit
# `name` but still want to inherit the parent project's config.
function _config_path()
    dir = dirname(Pkg.API.project().path)
    while true
        toml = joinpath(dir, "Project.toml")
        if isfile(toml)
            proj = Pkg.TOML.parsefile(toml)
            if haskey(proj, "name")
                return expanduser("~/config_" * proj["name"] * ".ini")
            end
        end
        parent = dirname(dir)
        parent == dir && error(
            "No named Project.toml found in any parent directory"
        )
        dir = parent
    end
end

function read_config()
    # Parse the file
    conf = ConfParser.ConfParse(_config_path())
    ConfParser.parse_conf!(conf)

    data = conf._data  # raw dictionary: Dict{String,Dict{String,Vector{String}}}

    # Prepare a clean dict with unwrapped strings
    clean = Dict{String, Dict{String, String}}()

    # iterate through sections and keys
    for (section, kvs) in data
        secdict = Dict{String,String}()
        for (key, valvec) in kvs
            # unwrap vector (ConfParser always returns Vector{String})
            valstr = _resolve(valvec[1], data)
            secdict[key] = valstr
        end
        clean[section] = secdict
    end

    return clean
end

# Recursive helper to resolve ${section:key} references
function _resolve(val::AbstractString, conf::Dict{Any,Any})
    # Look for ${section:key} patterns
    for m in eachmatch(r"\$\{([^:}]+):([^}]+)\}", val)
        sec = m.captures[1]
        key = m.captures[2]

        # retrieve referenced value and recurse
        repl = _resolve(conf[sec][key][1], conf)  # unwrap vector

        # replace in string
        val = replace(val, m.match => repl)
    end
    return val
end

end # module

