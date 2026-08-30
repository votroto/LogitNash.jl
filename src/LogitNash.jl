module LogitNash

include("utils.jl")
include("coordinates.jl")
include("kernels.jl")
include("predict.jl")
include("correct.jl")
include("tracker.jl")

export solve

end