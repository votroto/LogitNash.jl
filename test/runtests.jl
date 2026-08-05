using Test
using LogitNash
using LinearAlgebra

include("api.jl")

@testset "Unit tests" begin
    include("encoding.jl")
end

@testset "E2E game solving" begin
    include("kernels.jl")
    include("e2e.jl")
    include("nasty_games.jl")
end
