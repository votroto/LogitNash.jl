using Test
using LinearAlgebra

# --- Generative Helper Functions ---
# Each generator now returns a single N-dimensional tensor.

function gen_nearly_singular(dims; kappa=1e16)
    U = qr(randn(dims[1], dims[1])).Q
    V = qr(randn(dims[2], dims[2])).Q
    sigma = exp.(range(0, -log(kappa), length=min(dims[1], dims[2])))
    A2 = U[:, 1:length(sigma)] * Diagonal(sigma) * V[:, 1:length(sigma)]'
    return repeat(A2, 1, 1, dims[3:end]...)
end

function gen_singular_columns(dims)
    A = randn(dims...)
    selectdim(A, 2, 2) .= selectdim(A, 2, 1)
    return A
end

function gen_zero_row(dims)
    A = randn(dims...)
    selectdim(A, 1, size(A, 1)) .= 0.0
    return A
end

function gen_jordan_block(dims; lambda=1.0)
    A = fill(0.0, dims...)
    n = min(dims[1], dims[2])
    for i in 1:n
        selectdim(selectdim(A, 1, i), 1, i) .= lambda
        if i < n
            selectdim(selectdim(A, 1, i), 1, i+1) .= 1.0
        end
    end
    return A
end

function gen_extreme_scales(dims)
    scales = [1e-300, 1e-150, 1e-50, 1.0, 1e50, 1e150, 1e300]
    A = randn(dims...)
    for i in 1:min(dims[2], length(scales))
        selectdim(A, 2, i) .*= scales[i]
    end
    return A
end

function gen_subnormals(dims)
    A = randn(dims...)
    A[1] = nextfloat(0.0)
    A[2] = prevfloat(floatmin(Float64))
    return A
end

function gen_hilbert(dims)
    H2 = [1.0 / (i + j - 1) for i in 1:dims[1], j in 1:dims[2]]
    return repeat(H2, 1, 1, dims[3:end]...)
end

function gen_constant_payoffs(dims; c=42.0)
    return fill(c, dims...)
end

function gen_highly_sparse(dims; sparsity=0.95)
    A = randn(dims...)
    A[rand(length(A)) .< sparsity] .= 0.0
    return A
end

function gen_strictly_negative(dims; shift=-1e6)
    return -abs.(randn(dims...)) .+ shift
end

function gen_single_outlier(dims; outlier_val=1e15)
    A = randn(dims...)
    A[rand(1:length(A))] = outlier_val
    return A
end

# --- Robustness Test Runner ---

function assert_nash_handles_gracefully(dimensions, generator; samples=5)
    for _ in 1:samples
        payoffs = ntuple(_ -> generator(dimensions), length(dimensions))
        ne, status = @test_nowarn nash(payoffs; stop_iters=500, stop_t=1e6, stop_eps=1e-6)
        normal_exit = !status.stall && (status.t >= 1e6 || status.iteration >= 500 || status.regret <= 1e-6)
        @test normal_exit
        if !normal_exit
            @error "$status"
        end
    end
end


# Multi-order Magnitude Spans: singular exceptions
# Strictly Negative Payoffs: stall
# Single Massive Outlier: stall

# --- Main Test Suite ---

@testset "Nasty games finish gracefully" begin
    # 2-Player Matrix Cases
    dims2 = (6, 6)
    @testset "2P Nearly Singular" assert_nash_handles_gracefully(dims2, gen_nearly_singular)
    @testset "2P Identical Columns" assert_nash_handles_gracefully(dims2, gen_singular_columns)
    @testset "2P Zero Payoff Vector" assert_nash_handles_gracefully(dims2, gen_zero_row)
    @testset "2P Defective Jordan Structure" assert_nash_handles_gracefully(dims2, gen_jordan_block)
    @testset "2P Multi-order Magnitude Spans" assert_nash_handles_gracefully(dims2, gen_extreme_scales)
    @testset "2P Subnormal Float Densities" assert_nash_handles_gracefully(dims2, gen_subnormals)
    @testset "2P Ill-conditioned Hilbert Matrix" assert_nash_handles_gracefully(dims2, gen_hilbert)
    @testset "2P Constant / Zero Gradient" assert_nash_handles_gracefully(dims2, gen_constant_payoffs)
    @testset "2P Highly Sparse (95% zeros)" assert_nash_handles_gracefully(dims2, gen_highly_sparse)
    @testset "2P Strictly Negative Payoffs" assert_nash_handles_gracefully(dims2, gen_strictly_negative)
    @testset "2P Single Massive Outlier" assert_nash_handles_gracefully(dims2, gen_single_outlier)

    # 3-Player Tensor Cases
    dims3 = (4, 4, 4)
    @testset "3P Tensor Ill-conditioned" assert_nash_handles_gracefully(dims3, gen_nearly_singular)
    @testset "3P Tensor Extreme Scale Spans" assert_nash_handles_gracefully(dims3, gen_extreme_scales)
    @testset "3P Constant Game" assert_nash_handles_gracefully(dims3, gen_constant_payoffs)
    @testset "3P Highly Sparse (95% zeros)" assert_nash_handles_gracefully(dims3, gen_highly_sparse)
    @testset "3P Strictly Negative Payoffs" assert_nash_handles_gracefully(dims3, gen_strictly_negative)
    @testset "3P Single Massive Outlier" assert_nash_handles_gracefully(dims3, gen_single_outlier)
end