# julia --project=. -e 'using Pkg; Pkg.test(julia_args=["--code-coverage=coverage.info","--code-coverage=@src"])'
# genhtml coverage.info --output-directory /tmp/LogitNash_coverage

using Test
using LogitNash
using LinearAlgebra

# TODO: Add test to guarantee no debuging output forgotten on main!
# Simple games should return with empty stdout and stderr.
# (empty log with @info... too?)

include("api.jl")

@testset "Unit tests" begin
    include("encoding.jl")
end

@testset "E2E game solving" begin
    include("kernels.jl")
    include("e2e.jl")
    include("nasty_games.jl")
end
