module LogitNash

# Core
include("utils.jl")
include("coordinates.jl")
include("kernels.jl")
include("predict.jl")
include("correct.jl")
include("tracker.jl")

export solve

# Extensions
include("extensions/parse_nfg.jl")

end