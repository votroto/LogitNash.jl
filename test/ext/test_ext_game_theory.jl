using LogitNash, GameTheory
using Test

const GameTheoryExt = Base.get_extension(LogitNash, :GameTheoryExt)

struct MockGame{N} <: AbstractNormalFormGame{N,Float64}
    dims::NTuple{N,Int}
end

GameTheory.nums_actions(g::MockGame{N}) where {N} = g.dims

GameTheory.payoff_profile(::MockGame, a) = a

function test_extract_lazy()
    dims = (2, 3, 4)
    expected = ntuple(p -> map(i -> i[p], CartesianIndices(dims)), 3)

    g = MockGame(dims)
    actual = GameTheoryExt._extract_tensors(g)

    @test all(expected .== actual)
end

function test_extract_tensor()
    expected = ntuple(_ -> rand(2, 5, 4), 3)
    g = NormalFormGame(collect(zip(expected...)))
    actual = GameTheoryExt._extract_tensors(g)

    @test all(expected .== actual)
end

function test_one_randn_game()
    Us = ntuple(_ -> randn(2, 3, 4, 3, 2, 3), 6)

    g = NormalFormGame(collect(zip(Us...)))

    neg, stg = solve(g)
    ned, std = solve(Us)

    @test stg.stall == false && std.stall == false
    @test all(neg[p] ≈ ned[p] for p in 1:6)
end

@testset "Extension GameTheory.jl" begin
    test_extract_tensor()
    test_extract_lazy()
    test_one_randn_game()
end