# julia --project=. -e 'using Pkg; Pkg.test(julia_args=["--code-coverage=coverage.info","--code-coverage=@src"])'
# genhtml coverage.info --output-directory /tmp/LogitNash_coverage

using Test
using LogitNash
using LinearAlgebra

@testset "Basics" begin
    include("api.jl")
end

@testset "Unit tests" begin
    include("encoding.jl")
    include("kernels.jl")
end

@testset "E2E game solving" begin
    @test_nowarn redirect_stdout(stderr) do
        include("e2e.jl")
    end
    include("nasty_games.jl")
end
