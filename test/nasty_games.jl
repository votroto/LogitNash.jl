using Test
using LinearAlgebra

function gen_nearly_singular(dims; kappa=1e16)
    U = qr(randn(dims[1], dims[1])).Q
    V = qr(randn(dims[2], dims[2])).Q
    sigma = exp.(range(0, -log(kappa), length=min(dims[1], dims[2])))
    A2 = U[:, 1:length(sigma)] * Diagonal(sigma) * V[:, 1:length(sigma)]'
    return repeat(A2, 1, 1, dims[3:end]...)
end

function gen_duplicate_columns(dims)
    A = randn(dims...)
    selectdim(A, 2, 2) .= selectdim(A, 2, 1)
    return A
end

function gen_close_columns(dims)
    A = randn(dims...)
    selectdim(A, 2, 2) .*= 1e-6
    selectdim(A, 2, 2) .+= selectdim(A, 2, 1)
    return A
end

function gen_zero_row(dims)
    A = randn(dims...)
    selectdim(A, 1, size(A, 1)) .= 0.0
    return A
end

function gen_subnormals(dims)
    randn(dims...)*floatmin()
end

function gen_strictly_negative(dims; shift=-1e6)
    return -abs.(randn(dims...)) .+ shift
end

function gen_single_outlier(dims; outlier_val=1e15)
    A = randn(dims...)
    A[rand(1:length(A))] = outlier_val
    return A
end

# Scale raw matrix into roughly [-2.0, 0.0] trying to preserve the mantissa.
function matrix_rescale(arr::AbstractArray{T}) where T
    max_val = maximum(arr)
    min_val = minimum(arr)
    range_val = max_val - min_val
    if range_val <= eps(T)
        return arr
    end
    scale_exp = exponent(range_val)
    brr = similar(arr)
    for i in eachindex(brr)
        brr[i] = ldexp.(arr[i] .- max_val, -scale_exp)
    end
    return brr
end

function test_nash_stops_successfully(dimensions, generator; samples=5)
    for _ in 1:samples
        payoffs = ntuple(_ -> generator(dimensions), length(dimensions))
        _, status = @test_nowarn nash(payoffs; stop_iters=500, stop_lambda=1e6, stop_eps=1e-6)
        normal_exit = !status.stall && (status.lambda >= 1e6 || status.iteration >= 500 || status.regret <= 1e-6)
        @test normal_exit
        if !normal_exit
            @error "$status"
        end
    end
end

function test_unscaled_games_succeed_directly_or_after_scaling(dimensions, generator; attempts=20)
    for _ in 1:attempts
        payoffs_nasty = ntuple(_ -> generator(dimensions), length(dimensions))
        _, status_nasty = @test_nowarn nash(payoffs_nasty; stop_iters=500, stop_lambda=1e6, stop_eps=1e-6)
        if !status_nasty.stall
            continue
        end
        # Found a stalling game
        payoffs_scaled = matrix_rescale.(payoffs_nasty)
        _, status_scaled = nash(payoffs_scaled; stop_iters=500, stop_lambda=1e6, stop_eps=1e-6)
        @test !status_scaled.stall
        return
    end
    @test false skip=true
end

@testset "Nasty games tests" begin
    dims2 = (6, 6)
    dims3 = (4, 4, 4)
    @testset "2P Nearly Singular" test_nash_stops_successfully(dims2, gen_nearly_singular)
    @testset "2P Identical Columns" test_nash_stops_successfully(dims2, gen_duplicate_columns)
    @testset "2P Close Columns" test_nash_stops_successfully(dims2, gen_close_columns)
    @testset "2P Zero Payoff Vector" test_nash_stops_successfully(dims2, gen_zero_row)
    @testset "2P Subnormal Floats" test_nash_stops_successfully(dims2, gen_subnormals)
    @testset "3P Tensor Ill-conditioned" test_nash_stops_successfully(dims3, gen_nearly_singular)
end
@testset "Scaling tests" begin
    dims2 = (6, 6)
    @testset "Scaling Offset" test_unscaled_games_succeed_directly_or_after_scaling(dims2, gen_strictly_negative)
    @testset "Scaling Outlier" test_unscaled_games_succeed_directly_or_after_scaling(dims2, gen_single_outlier)
end