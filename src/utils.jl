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

function max_deviation_incentive(ubar::NTuple{N}, pi::NTuple{N}) where N
    sum(maximum(ubar[p]) - dot(ubar[p], pi[p]) for p in 1:N)
end

function lu_det_sign_rcond_heur(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    s = 1.0
    min_d = Inf
    max_d = 1e-18 # Add a tiny epsilon to prevent divide-by-zero
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
    rcond = min_d / max_d
    return (rcond >= 1e-5) ? s : 0.0
end

function fast_lu!(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    A, ipiv, info = LinearAlgebra.LAPACK.getrf!(A, ipiv)
    return info
end

function shift_and_insert!(v::AbstractVector, src::Int, dest::Int, val)
    if src < dest
        # Shift elements left to close the gap
        @inbounds for i in src:(dest-1)
            v[i] = v[i+1]
        end
    elseif src > dest
        # Shift elements right to close the gap
        @inbounds for i in src:-1:(dest+1)
            v[i] = v[i-1]
        end
    end
    @inbounds v[dest] = val
    return v
end
