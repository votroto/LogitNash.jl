using LinearAlgebra: BlasInt

function validate_game(utils::NTuple{N, Array{R}}) where {N,R}
    if N <= 1
        throw(ArgumentError("A normal-form game must have at least 2 players; got N = $N."))
    end

    if any(isempty, utils)
        throw(ArgumentError("Utility matrices cannot be empty."))
    end
    if !allequal(size, utils)
        throw(DimensionMismatch("All utility matrices must have matching sizes. Received sizes: $(map(size, utils))"))
    end
    for U in utils
        for u in U
            if !isfinite(u)
                throw(ArgumentError("Utility matrices contain Infs or NaNs"))
            end
        end
    end
    if Base.promote_op(*, R, Float64) != Float64
        @warn "Precision may be lost. $R does not promote to Float64."
    end
    return true
end

function splitviews(x::AbstractVector, js::NTuple{N,Int}) where {N}
    offs = cumsum((0, js...))
    ntuple(i -> @view(x[(offs[i]+1):offs[i+1]]), N)
end

function redlograt_to_prob!(y::AbstractVector{F}, x::AbstractVector{F}, active::Vector{Int}, ref::Int) where F
    c = maximum(x, init=zero(F))
    denom = zero(F)

    @inbounds for i in eachindex(x)
        a = active[i]
        v = exp(x[i] - c)
        y[a] = v
        denom += v
    end

    y[ref] = exp(-c)
    denom += y[ref]

    @inbounds for i in eachindex(y)
        y[i] = y[i] / denom
    end

    return y
end

function max_deviation_incentive(ubar::NTuple{N}, pi::NTuple{N}) where N
    sum(maximum(ubar[p]) - dot(ubar[p], pi[p]) for p in 1:N)
end

function lu_det_sign_rcond_heur(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    s = 1.0
    min_d = Inf
    max_d = 0.0
    @inbounds for i in axes(A, 1)
        val = A[i, i]
        if val < 0.0
            s = -s
        end
        if ipiv[i] != i
            s = -s
        end

        abs_val = abs(val)
        min_d = min(min_d, abs_val)
        max_d = max(max_d, abs_val)
    end
    # Add a tiny epsilon to prevent divide-by-zero
    rcond = min_d / (max_d + 1e-18)
    return (rcond >= 1e-5) ? s : 0.0
end

function fast_lu!(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    A, ipiv, info = LinearAlgebra.LAPACK.getrf!(A, ipiv)
    return info
end

function uniform_xprofile(Us)
    nx = sum(size(Us[i], i) - 1 for i in eachindex(Us))
    zeros(nx)
end
