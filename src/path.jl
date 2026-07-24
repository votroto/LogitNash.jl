using LinearAlgebra
using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: liblapack

function validate_game(utils::NTuple{N,AbstractArray{F,N}}) where {F,N}
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
    return true
end

function max_deviation_incentive(ubar::NTuple{N}, pi::NTuple{N}) where N
    maximum(maximum(ubar[p]) - dot(ubar[p], pi[p]) for p in 1:N)
end

function fast_lu!(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    A, ipiv, info = LinearAlgebra.LAPACK.getrf!(A, ipiv)

    if info > 0
        throw(SingularException(info))
    end
    return A
end

_zero_nested!(x::AbstractArray) = fill!(x, zero(eltype(x)))
function _zero_nested!(t::Tuple)
    for ti in t
        _zero_nested!(ti)
    end
end

function make_hc_workspace(x_template::Vector{Float64}, utils::NTuple{N}) where {N}
    T = eltype(x_template)
    n = length(x_template)
    rsize = sum(size(first(utils), i) - 1 for i in eachindex(utils))

    pi = ntuple(i -> Vector{T}(undef, size(utils[i], i)), Val(N))
    res = Vector{T}(undef, rsize)
    ubar = ntuple(i -> zeros(T, size(utils[i], i)), Val(N))
    dudpi = ntuple(p -> ntuple(q -> zeros(T, size(utils[p], p), size(utils[p], q)), Val(N)), Val(N))

    J_aug = zeros(T, n + 1, n + 1)
    Fx = view(J_aug, 1:n, 1:n)
    Ft = view(J_aug, 1:n, n + 1)
    ipiv = Vector{BlasInt}(undef, n + 1)

    rhs_aug = Vector{T}(undef, n + 1)

    xpred = Vector{T}(undef, n)
    x_diff = Vector{T}(undef, n)
    dx_step = Vector{T}(undef, n)
    x_nxt = Vector{Float64}(undef, n)

    det_sign = Float64[0.0]

    return (; pi, res, ubar, dudpi, J_aug, Fx, Ft, ipiv, rhs_aug, xpred, x_diff, dx_step, x_nxt, det_sign)
end

function predict!(
    dx_out::Vector{Float64},
    x::Vector{Float64},
    t::Float64,
    lastdx::Vector{Float64},
    lastdt::Float64,
    utils::NTuple{N},
    ws
) where {N}
    _zero_nested!(ws.ubar)
    _zero_nested!(ws.dudpi)

    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
    jacobian_x!(ws.Fx, ws.pi, t, ws.dudpi, utils)

    unilateral_deviations!(ws.ubar, utils, ws.pi)
    jacobian_t!(ws.Ft, ws.ubar, mu, utils)

    n = length(x)
    @inbounds for i in 1:n
        ws.J_aug[end, i] = lastdx[i]
    end
    ws.J_aug[end, end] = lastdt

    fill!(ws.rhs_aug, 0.0)
    ws.rhs_aug[end] = 1.0

    fast_lu!(ws.J_aug, ws.ipiv)

    cur_det_sign = 1.0
    n_aug = n + 1
    @inbounds for i in 1:n_aug
        if ws.J_aug[i, i] < 0.0
            cur_det_sign = -cur_det_sign
        end
        if ws.ipiv[i] != i
            cur_det_sign = -cur_det_sign
        end
    end
    if ws.det_sign[1] == 0.0
        ws.det_sign[1] = cur_det_sign
    end

    LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

    norm_factor = 1.0 / sqrt(dot(ws.rhs_aug, ws.rhs_aug))

    @inbounds for i in 1:n
        dx_out[i] = ws.rhs_aug[i] * norm_factor
    end
    dtds = ws.rhs_aug[end] * norm_factor

    return dx_out, dtds
end

function correct!(
    xlast::Vector{Float64},
    tlast::Float64,
    dx::Vector{Float64},
    dt::Float64,
    ds::Float64,
    utils::NTuple{N},
    ws;
    iters::Int=3,
    abs_tol::Float64=1e-6,
    rel_tol::Float64=1e-12,
    bifurcation_ds_tol::Float64=1e-5
) where {N}
    @. ws.xpred = xlast + ds * dx
    tpred = tlast + ds * dt

    copyto!(ws.x_nxt, ws.xpred)
    t_out = tpred

    n = length(dx)
    i = 0

    # Track sign locally to prevent corrupting workspace on rejected steps
    local_det_sign = ws.det_sign[1]

    while true
        fill!(ws.res, 0.0)
        _zero_nested!(ws.ubar)

        mu = splitviews(ws.x_nxt, size(first(utils)) .- 1)
        redlograt_to_prob!.(ws.pi, mu)

        unilateral_deviations!(ws.ubar, utils, ws.pi)
        residual!(ws.res, mu, ws.ubar, ws.x_nxt, t_out, utils)

        @. ws.x_diff = ws.x_nxt - ws.xpred
        r_con = dot(ws.x_diff, dx) + (t_out - tpred) * dt

        if dot(ws.res, ws.res) + r_con^2 < abs_tol^2
            # Commit sign change on early exit success
            ws.det_sign[1] = local_det_sign
            return true, ws.x_nxt, t_out
        end

        _zero_nested!(ws.dudpi)

        unilateral_derivatives!(ws.dudpi, utils, ws.pi)
        jacobian_x!(ws.Fx, ws.pi, t_out, ws.dudpi, utils)
        jacobian_t!(ws.Ft, ws.ubar, mu, utils)

        @inbounds for j in 1:n
            ws.J_aug[end, j] = dx[j]
        end
        ws.J_aug[end, end] = dt

        @inbounds for j in 1:n
            ws.rhs_aug[j] = -ws.res[j]
        end
        ws.rhs_aug[end] = -r_con

        fast_lu!(ws.J_aug, ws.ipiv)

        # Fold Jump vs Bifurcation Detection ---
        cur_det_sign = 1.0
        n_aug = n + 1
        @inbounds for j in 1:n_aug
            if ws.J_aug[j, j] < 0.0
                cur_det_sign = -cur_det_sign
            end
            if ws.ipiv[j] != j
                cur_det_sign = -cur_det_sign
            end
        end

        # Distinguish between jumping a fold (large ds) and hitting a bifurcation (tiny ds)
        if local_det_sign != 0.0 && cur_det_sign != local_det_sign
            if ds > bifurcation_ds_tol
                # We likely jumped a tight fold. Reject and halve ds.
                return false, ws.x_nxt, t_out
            else
                # ds is tiny; this is a structural bifurcation. Accept the sign flip.
                local_det_sign = cur_det_sign
            end
        end

        LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

        dt_step = ws.rhs_aug[end]
        step_norm_sq = dt_step^2

        @inbounds for j in 1:n
            dx_step_j = ws.rhs_aug[j]
            ws.dx_step[j] = dx_step_j
            step_norm_sq += dx_step_j^2
        end

        step_norm = sqrt(step_norm_sq)
        val_norm = sqrt(dot(ws.x_nxt, ws.x_nxt) + t_out^2)

        if step_norm < rel_tol * val_norm
            @. ws.x_nxt += ws.dx_step
            t_out += dt_step

            dist_sq = (t_out - tpred)^2
            @inbounds for j in 1:n
                dist_sq += (ws.x_nxt[j] - ws.xpred[j])^2
            end
            if dist_sq > (2.0 * ds)^2
                return false, ws.x_nxt, t_out
            end

            # Step successfully converged and validated. Commit the sign lock.
            ws.det_sign[1] = local_det_sign
            return true, ws.x_nxt, t_out
        end

        if i >= iters
            return false, ws.x_nxt, t_out
        end

        @. ws.x_nxt += ws.dx_step
        t_out += dt_step
        i += 1
    end
end


"""
    nash(utils::NTuple{N,AbstractArray{Float64,N}}; stop_iters::Int=1000, stop_t::Float64=1e6, stop_eps::Float64=1e-6) where {N}

Compute an epsilon Nash equilibrium of an N-player game by tracing a logit equilibrium path from t=0 to infinity.
"""
function nash(
    utils::NTuple{N,AbstractArray{Float64,N}};
    stop_iters::Int=1000,
    stop_t::Float64=1e6,
    stop_eps::Float64=1e-6
) where {N}
    validate_game(utils)

    x = uniform_xprofile(utils)
    t = 0.0
    dx = zero(x)
    dt = 1.0
    ds = 0.01
    iteration = 0
    successes_in_row = 0
    regret = NaN
    stall = false

    ws = make_hc_workspace(x, utils)

    while t <= stop_t && iteration <= stop_iters && !stall
        dx, dt = predict!(dx, x, t, dx, dt, utils, ws)
        regret = max_deviation_incentive(ws.ubar, ws.pi)

        if (regret <= stop_eps)
            break
        end

        corrected = false
        t_new = t
        while !corrected
            corrected, x_new, t_new = correct!(x, t, dx, dt, ds, utils, ws)

            if !corrected
                ds /= 2
                successes_in_row = 0
                if ds <= 1e-7
                    # Progress along path stalled!
                    stall = true
                    break
                end
            else
                copyto!(x, x_new)
                t = t_new

                successes_in_row += 1
                if successes_in_row >= 5
                    successes_in_row = 0
                    ds *= 2
                end

                iteration += 1
            end
        end

    end

    return ws.pi, (; t, iteration, regret, stall)
end
