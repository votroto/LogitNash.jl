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

function failing_strictly_negative()
    An = [-1.0000007089068863e6 -1.0000002648376371e6 -1.0000010692871211e6 -1.0000014882896636e6; -1.0000000679257057e6 -1.0000008579807434e6 -1.0000000201195937e6 -1.0000011301480398e6; -1.0000001840893195e6 -1.0000009341092664e6 -1.0000015452264716e6 -1.0000002565338324e6; -1.0000000479744782e6 -1.0000002368189936e6 -1.000000623370072e6 -1.0000004211651277e6]
    Bn = [-1.0000003575490277e6 -1.0000004524285814e6 -1.0000002667256924e6 -1.0000007409103559e6; -1.0000005867706521e6 -1.0000013466278106e6 -1.0000003635754004e6 -1.000000004770767e6; -1.0000002032162965e6 -1.0000019414389707e6 -1.0000001141638706e6 -1.0000015677850158e6; -1.0000009417161624e6 -1.0000001657964108e6 -1.0000002681505332e6 -1.000001140079594e6]
    return (An, Bn)
end

function failing_single_outlier()
    Ao = [-0.7089068862280788 0.2648376370963854 -1.0692871210059594 -1.4882896635378573; -0.06792570574891812 -0.8579807433624448 -0.020119593750427267 -1.1301480397751438; -0.18408931948988505 -0.9341092663504892 1.0e15 -0.25653383238294997; 0.047974478231682514 -0.23681899356161887 0.6233700719723749 0.4211651277129798]
    Bo = [-0.5867706520714414 1.3466278106274674 1.0e15 0.004770766980493883; 0.2032162965517312 -1.9414389707325737 -1.123377312657212 -1.5677850158181024; -0.9417161623330623 0.1657964107079637 -0.26815053324248866 1.1400795939357375; 0.45242858143912607 -1.6672817303457028 -0.7409103559460443 -0.4254523226698724]
    return (Ao, Bo)
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

function test_nash_stops_successfully(dimensions, generator; samples=5, rescale=false)
    scaler = rescale ? matrix_rescale : identity
    for _ in 1:samples
        payoffs = ntuple(_ -> scaler(generator(dimensions)), length(dimensions))
        ne, status = @test_nowarn nash(payoffs; stop_iters=500, stop_t=1e6, stop_eps=1e-6)
        normal_exit = !status.stall && (status.t >= 1e6 || status.iteration >= 500 || status.regret <= 1e-6)
        @test normal_exit
        if !normal_exit
            @error "$status"
        end
    end
end

function test_nash_stalls_gracefully(payoffs)
    _, status = @test_nowarn nash(payoffs; stop_iters=500, stop_t=1e6, stop_eps=1e-6)
    @test status.stall
end

function test_nash_does_not_stall_after_rescaling(payoffs)
    _, status = @test_nowarn nash(matrix_rescale.(payoffs); stop_iters=500, stop_t=1e6, stop_eps=1e-6)
    @test !status.stall
end

dims2 = (6, 6)
dims3 = (4, 4, 4)

@testset "Nasty 2p games finish successfully" begin
    @testset "2P Nearly Singular" test_nash_stops_successfully(dims2, gen_nearly_singular)
    @testset "2P Identical Columns" test_nash_stops_successfully(dims2, gen_duplicate_columns)
    @testset "2P Close Columns" test_nash_stops_successfully(dims2, gen_close_columns)
    @testset "2P Zero Payoff Vector" test_nash_stops_successfully(dims2, gen_zero_row)
    @testset "2P Subnormal Float Densities" test_nash_stops_successfully(dims2, gen_subnormals)
end

@testset "Nasty 3p games finish successfully" begin
    @testset "3P Tensor Ill-conditioned" test_nash_stops_successfully(dims3, gen_nearly_singular)
end

@testset "Rescaled games finish successfully" begin
    @testset "2P Strictly Negative Payoffs" test_nash_stops_successfully(dims2, gen_strictly_negative; rescale=true)
    @testset "2P Single Massive Outlier" test_nash_stops_successfully(dims2, gen_single_outlier; rescale=true)
    @testset "3P Strictly Negative Payoffs" test_nash_stops_successfully(dims3, gen_strictly_negative; rescale=true)
    @testset "3P Single Massive Outlier" test_nash_stops_successfully(dims3, gen_single_outlier; rescale=true)
end

@testset "Scaling solves stalls" begin
    @testset "2P Strictly Negative Payoffs" test_nash_stalls_gracefully(failing_single_outlier())
    @testset "2P Single Massive Outlier" test_nash_stalls_gracefully(failing_strictly_negative())
    @testset "2P Strictly Negative Payoffs" test_nash_does_not_stall_after_rescaling(failing_single_outlier())
    @testset "2P Single Massive Outlier" test_nash_does_not_stall_after_rescaling(failing_strictly_negative())
end