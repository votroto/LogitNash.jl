using LinearAlgebra
using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: liblapack

@enum StepStatus begin
    STATUS_SUCCESS
    STATUS_MAX_ITERS
    STATUS_SINGULAR
    STATUS_LARGE_DISTANCE
    STATUS_JUMP
end

function validate_game(utils::NTuple{N,AbstractArray}) where N
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

function make_hc_workspace(x_template::Vector{Float64}, dims::NTuple{N}) where {N}
    T = eltype(x_template)
    n = length(x_template)
    rsize = sum(dims[i] - 1 for i in 1:N)

    pi = ntuple(i -> Vector{T}(undef, dims[i]), Val(N))
    res = Vector{T}(undef, rsize)
    ubar = ntuple(i -> zeros(T, dims[i]), Val(N))
        dudpi = ntuple(p -> ntuple(q -> zeros(T, dims[p], dims[q]), Val(N)), Val(N))
    # q=1 transposed for improved memory access pattern
#    dudpi = ntuple(p -> ntuple(q -> (q == 1) ? zeros(dims[q], dims[p]) : zeros(dims[p], dims[q]), Val(N)), Val(N))

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

function lu_det_sign(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    s = 1.0
    @inbounds for i in axes(A, 1)
        if A[i, i] < 0.0
            s = -s
        end
        if ipiv[i] != i
            s = -s
        end
    end
    return s
end

function fast_lu!(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    A, ipiv, info = LinearAlgebra.LAPACK.getrf!(A, ipiv)
    return info
end

function predict_step!(xpred::Vector{Float64}, x::Vector{Float64}, t::Float64, dx::Vector{Float64}, dt::Float64, ds::Float64)
    @. xpred = x + ds * dx
    tpred = t + ds * dt
    return xpred, tpred
end

function predict_direction!(dx::Vector{Float64}, dt::Float64, ws)
    n = length(dx)
    fill!(ws.rhs_aug, 0.0)
    ws.rhs_aug[end] = 1.0

    LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

    norm_factor = 1.0 / sqrt(dot(ws.rhs_aug, ws.rhs_aug))
    @inbounds for i in 1:n
        dx[i] = ws.rhs_aug[i] * norm_factor
    end
    dt = ws.rhs_aug[end] * norm_factor

    return dx, dt
end

function predict_init!(x::Vector{Float64}, t::Float64, utils::NTuple{N}, ws) where {N}
    n = length(x)
    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives_fast2!(ws.dudpi, utils, ws.pi)
    unilateral_deviations_from_derivatives2!(ws.ubar, ws.dudpi, ws.pi)

    jacobian_x3!(ws.Fx, ws.pi, t, ws.dudpi, utils)
    jacobian_t!(ws.Ft, ws.ubar, mu, utils)

    @inbounds for i in 1:n
        ws.J_aug[end, i] = 0.0
    end
    ws.J_aug[end, end] = 1.0

    info = fast_lu!(ws.J_aug, ws.ipiv)
    if info > 0
        throw(SingularException(info))
    end

    ws.det_sign[1] = lu_det_sign(ws.J_aug, ws.ipiv)

    nothing
end
using Printf
function correct!(
    x_nxt::Vector{Float64},
    xpred::Vector{Float64},
    tpred::Float64,
    dx::Vector{Float64},
    dt::Float64,
    utils::NTuple{N},
    ws;
    max_iters::Int=3,
    abs_tol::Float64=1e-6,
    rel_tol::Float64=1e-12
) where {N}
    copyto!(x_nxt, xpred)
    t_out = tpred
    n = length(x_nxt)

    for i in 0:max_iters
        mu = splitviews(x_nxt, size(first(utils)) .- 1)
         redlograt_to_prob!.(ws.pi, mu)
       unilateral_deviations!(ws.ubar, utils, ws.pi)
        residual!(ws.res, mu, ws.ubar, x_nxt, t_out, utils)

        @. ws.x_diff = x_nxt - xpred
        r_con = dot(ws.x_diff, dx) + (t_out - tpred) * dt

        # Early exit avoids computing derivatives/factorization
        if dot(ws.res, ws.res) + r_con^2 < abs_tol^2
            return STATUS_SUCCESS, i, x_nxt, t_out
        end

        if i == max_iters
            return STATUS_MAX_ITERS, max_iters, x_nxt, t_out
        end

        unilateral_derivatives_fast2_cleaner!(ws.dudpi, utils, ws.pi)
        jacobian_x3!(ws.Fx, ws.pi, t_out, ws.dudpi, utils)
        jacobian_t!(ws.Ft, ws.ubar, mu, utils)

        @inbounds for j in 1:n
            ws.J_aug[end, j] = dx[j]
        end
        ws.J_aug[end, end] = dt

        @inbounds for j in 1:n
            ws.rhs_aug[j] = -ws.res[j]
        end
        ws.rhs_aug[end] = -r_con

        #println((@sprintf("%9.5f ",s) for s in ws.Fx)...)

        info = fast_lu!(ws.J_aug, ws.ipiv)
        if info > 0
            return STATUS_SINGULAR, i, x_nxt, t_out
        end

        LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

        dt_step = ws.rhs_aug[end]
        step_norm_sq = dt_step^2

        @inbounds for j in 1:n
            ws.dx_step[j] = ws.rhs_aug[j]
            step_norm_sq += ws.dx_step[j]^2
        end

        step_norm = sqrt(step_norm_sq)
        val_norm = sqrt(dot(x_nxt, x_nxt) + t_out^2)

        @. x_nxt += ws.dx_step
        t_out += dt_step

        if step_norm < rel_tol * val_norm
            return STATUS_SUCCESS, i + 1, x_nxt, t_out
        end
    end

    return STATUS_MAX_ITERS, max_iters, x_nxt, t_out
end

function validate_step!(
    status::StepStatus,
    iters::Int,
    x_nxt::Vector{Float64},
    t_new::Float64,
    xpred::Vector{Float64},
    tpred::Float64,
    ds::Float64,
    ws
)
    if status != STATUS_SUCCESS
        return status
    end

    if iters == 0
        return STATUS_SUCCESS
    end

    dist_sq = (t_new - tpred)^2
    @inbounds for j in 1:length(x_nxt)
        dist_sq += (x_nxt[j] - xpred[j])^2
    end

    if dist_sq > (2.0 * ds)^2
        return STATUS_LARGE_DISTANCE
    end

    cur_det_sign = lu_det_sign(ws.J_aug, ws.ipiv)
    if ws.det_sign[1] != 0.0 && cur_det_sign != ws.det_sign[1]
        if dist_sq > (0.5 * ds)^2
            return STATUS_JUMP
        else
            ws.det_sign[1] = cur_det_sign
        end
    end

    return STATUS_SUCCESS
end

"""
    nash(utils::NTuple{N,AbstractArray{Float64,N}}; stop_iters::Int=1000, stop_t::Float64=1e6, stop_eps::Float64=1e-6) where {N}
Compute an epsilon Nash equilibrium of an N-player game by tracing a logit equilibrium path from t=0 to infinity.
"""
function nash(
    utils::NTuple{N,AbstractArray{Float64,N}};
    stop_iters::Int=1000,
    stop_t::Float64=1e6, # TODO: rename to precision
    stop_eps::Float64=1e-6  # TODO: rename to regret
) where {N}
    validate_game(utils)

    x = uniform_xprofile(utils)
    t = 0.0
    dx = zero(x)
    dt = 1.0
    ds = 0.01

    ws = make_hc_workspace(x, size(first(utils)))
    predict_init!(x, t, utils, ws)

    iteration = 0
    successes_in_row = 0
    regret = NaN
    stall = false

    while t <= stop_t && iteration <= stop_iters && !stall
        dx, dt = predict_direction!(dx, dt, ws)

        regret = max_deviation_incentive(ws.ubar, ws.pi)
        if regret <= stop_eps
            break
        end

        while true
            xpred, tpred = predict_step!(ws.xpred, x, t, dx, dt, ds)
            corr_status, iters, x_nxt, t_nxt = correct!(ws.x_nxt, xpred, tpred, dx, dt, utils, ws)
#print(iters, " ")
            val_status = validate_step!(corr_status, iters, x_nxt, t_nxt, xpred, tpred, ds, ws)

            if val_status == STATUS_SUCCESS
                copyto!(x, x_nxt)
                t = t_nxt
                break
            else
                ds /= 2.0
                successes_in_row = 0
                if ds <= 1e-7
                    stall = true
                    break
                end
            end
        end

        successes_in_row += 1
        if successes_in_row >= 5
            successes_in_row = 0
            ds *= 2.0
        end
        iteration += 1
    end
println()
    return ws.pi, (; t, iteration, regret, stall)
end