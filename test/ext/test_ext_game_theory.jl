using LogitNash, GameTheory
using Test

function test_one_randn_game()
    Us = ntuple(_ -> randn(2, 3, 4, 3, 2, 3), 6)

    g = NormalFormGame(collect(zip(Us...)))

    neg, stg = solve(g)
    ned, std = solve(Us)

    @test stg.stall == false && std.stall == false
    @test all(neg[p] ≈ ned[p] for p in 1:6)
end

@testset "Extension GameTheory" begin
    test_one_randn_game()
end