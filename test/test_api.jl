function single_player_game_is_rejected()
    A = rand(3)
    @test_throws ArgumentError solve((A,))
end

function empty_games_are_rejected()
    payoffs302 = (rand(3,0,2), rand(3,0,2), rand(3,0,2))
    payoffs00 = (randn(0,0), rand(0,0))
    @test_throws ArgumentError solve(payoffs302)
    @test_throws ArgumentError solve(payoffs00)
end

function dimensions_agree_with_playercount()
    payoffs = (rand(3,1,2), rand(3,1,2))
    @test_throws MethodError solve(payoffs)
end

function mismatched_dimensions_are_detected()
    payoffs = (rand(3,2), rand(2,3))
    @test_throws DimensionMismatch solve(payoffs)
end

function infs_nans_are_detected()
    payoffsInf = (rand(3,2), rand(3,2))
    payoffsInf[1][end] = Inf
    payoffsNaN = (rand(3,2), rand(3,2))
    payoffsNaN[1][end] = NaN
    @test_throws ArgumentError solve(payoffsInf)
    @test_throws ArgumentError solve(payoffsNaN)
end

@testset "API tests" begin
    single_player_game_is_rejected()
    mismatched_dimensions_are_detected()
    empty_games_are_rejected()
    dimensions_agree_with_playercount()
    infs_nans_are_detected()
end