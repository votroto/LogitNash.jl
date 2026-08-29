using LinearAlgebra
using LinearAlgebra: BlasInt

@enum StepStatus begin
    STATUS_SUCCESS
    STATUS_MAX_ITERS
    STATUS_SINGULAR
    STATUS_LARGE_DISTANCE
    STATUS_JUMP
end

function make_hc_workspace(x_template::Vector{Float64}, dims::NTuple{N}) where {N}
    n = length(x_template)
    rsize = sum(dims[i] - 1 for i in 1:N)

    pi = ntuple(i -> Vector{Float64}(undef, dims[i]), Val(N))
    res = Vector{Float64}(undef, rsize)
    ubar = ntuple(i -> zeros(dims[i]), Val(N))
    dudpi = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), Val(N)), Val(N))

    J_aug = zeros(n + 1, n + 1)
    Fx = view(J_aug, 1:n, 1:n)
    Ft = view(J_aug, 1:n, n + 1)
    ipiv = Vector{BlasInt}(undef, n + 1)

    rhs_aug = Vector{Float64}(undef, n + 1)

    x_pred = Vector{Float64}(undef, n)
    x_nxt = Vector{Float64}(undef, n)

    det_sign = Float64[1.0]

    # --- NEW: Tracking structures for pivoting ---
    refs = Int[dims[i] for i in 1:N]
    active_actions = [collect(1:(dims[i]-1)) for i in 1:N]

    offsets = zeros(Int, N)
    cur = 0
    for i in 1:N
        offsets[i] = cur
        cur += dims[i] - 1
    end

    return (; pi, res, ubar, dudpi, J_aug, Fx, Ft, ipiv, rhs_aug, x_pred, x_nxt, det_sign, refs, active_actions, offsets)
end

function pivot_reference!(x::Vector{Float64}, dx::Vector{Float64}, dx_old::Vector{Float64}, p::Int, new_ref_idx::Int, ws)
    offset = ws.offsets[p]
    old_ref = ws.refs[p]
    new_ref = ws.active_actions[p][new_ref_idx]

    val_x = x[offset+new_ref_idx]
    val_dx = dx[offset+new_ref_idx]
    val_dx_old = dx_old[offset+new_ref_idx]

    @inbounds for i in 1:length(ws.active_actions[p])
        idx = offset + i
        x[idx] -= val_x
        dx[idx] -= val_dx
        dx_old[idx] -= val_dx_old
    end

    x[offset+new_ref_idx] = -val_x
    dx[offset+new_ref_idx] = -val_dx
    dx_old[offset+new_ref_idx] = -val_dx_old

    ws.active_actions[p][new_ref_idx] = old_ref
    ws.refs[p] = new_ref
end

function update_predictor_jacobian!(x::Vector{Float64}, t::Float64, dx::Vector{Float64}, dt::Float64, utils::NTuple{N}, ws) where {N}
    mu = splitviews(x, size(first(utils)) .- 1)

    # --- UPDATED: Pass active actions and ref ---
    for p in 1:N
        redlograt_to_prob!(ws.pi[p], mu[p], ws.active_actions[p], ws.refs[p])
    end

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
    unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

    lambda = expm1(t)

    # --- UPDATED: Pass active actions and ref ---
    jacobian_x!(ws.Fx, ws.pi, lambda, ws.dudpi, utils, ws.active_actions, ws.refs)
    jacobian_t!(ws.Ft, ws.ubar, mu, lambda, ws.active_actions, ws.refs)

    @inbounds for j in eachindex(dx)
        ws.J_aug[end, j] = dx[j]
    end
    ws.J_aug[end, end] = dt

    fast_lu!(ws.J_aug, ws.ipiv)
end

function predict_direction!(dx::Vector{Float64}, ws)
    fill!(ws.rhs_aug, 0.0)
    ws.rhs_aug[end] = 1.0

    LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

    norm_factor = 1.0 / sqrt(dot(ws.rhs_aug, ws.rhs_aug))
    @inbounds for i in eachindex(dx)
        dx[i] = ws.rhs_aug[i] * norm_factor
    end
    dt = ws.rhs_aug[end] * norm_factor

    return dx, dt
end


function predict_step_curvature!(
    x_pred::Vector{Float64},
    x::Vector{Float64},
    t::Float64,
    dx::Vector{Float64},
    dt::Float64,
    dx_old::Vector{Float64},
    dt_old::Float64,
    ds::Float64,
    ds_old::Float64
)
    if ds_old <= 0.0
        @. x_pred = x + ds * dx
        t_pred = t + ds * dt
    else
        c_factor = (ds^2) / (2.0 * ds_old)
        @. x_pred = x + ds * dx + c_factor * (dx - dx_old)
        t_pred = t + ds * dt + c_factor * (dt - dt_old)
    end
    return x_pred, t_pred
end

function correct!(
    x_nxt::Vector{Float64},
    x_pred::Vector{Float64},
    t_pred::Float64,
    dx::Vector{Float64},
    dt::Float64,
    utils::NTuple{N},
    ws;
    max_iters::Int=3,
    abs_tol::Float64=1e-6,
    rel_tol::Float64=1e-11
) where {N}
    copyto!(x_nxt, x_pred)
    t_nxt = t_pred

    for i in 0:max_iters
        mu = splitviews(x_nxt, size(first(utils)) .- 1)
        for p in 1:N
            redlograt_to_prob!(ws.pi[p], mu[p], ws.active_actions[p], ws.refs[p])
        end

        unilateral_derivatives!(ws.dudpi, utils, ws.pi)
        unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

        lambda = expm1(t_nxt)

        residual!(ws.res, ws.ubar, mu, lambda, ws.active_actions, ws.refs)

        r_con = (t_nxt - t_pred) * dt
        @inbounds @simd for j in eachindex(dx)
            r_con += (x_nxt[j] - x_pred[j]) * dx[j]
        end

        if dot(ws.res, ws.res) + r_con^2 < abs_tol^2
            return STATUS_SUCCESS, x_nxt, t_nxt
        end

        if i == max_iters
            return STATUS_MAX_ITERS, x_nxt, t_nxt
        end

        jacobian_x!(ws.Fx, ws.pi, lambda, ws.dudpi, utils, ws.active_actions, ws.refs)
        jacobian_t!(ws.Ft, ws.ubar, mu, lambda, ws.active_actions, ws.refs)

        @inbounds for j in eachindex(dx)
            ws.J_aug[end, j] = dx[j]
        end
        ws.J_aug[end, end] = dt

        @inbounds for j in eachindex(ws.res)
            ws.rhs_aug[j] = -ws.res[j]
        end
        ws.rhs_aug[end] = -r_con


        info = fast_lu!(ws.J_aug, ws.ipiv)
        if info > 0
            return STATUS_SINGULAR, x_nxt, t_nxt
        end

        LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

        dt_step = ws.rhs_aug[end]
        step_norm_sq = dt_step^2

        @inbounds @simd for j in eachindex(x_nxt)
            step_j = ws.rhs_aug[j]
            step_norm_sq += step_j^2
            x_nxt[j] += step_j
        end

        step_norm = sqrt(step_norm_sq)
        val_norm = sqrt(dot(x_nxt, x_nxt) + t_nxt^2)

        t_nxt += dt_step

        # Catch wildly diverging Newton steps before they corrupt the Jacobian with Infs.
        if !isfinite(t_nxt) || t_nxt > 50.0 || t_nxt < -20.0 || any(!isfinite, x_nxt)
            return STATUS_LARGE_DISTANCE, x_nxt, t_nxt
        end

        if step_norm < rel_tol * val_norm
            return STATUS_SUCCESS, x_nxt, t_nxt
        end
    end

    return STATUS_MAX_ITERS, x_nxt, t_nxt
end

function validate_step!(
    status::StepStatus,
    x_next::Vector{Float64},
    t_next::Float64,
    x_pred::Vector{Float64},
    t_pred::Float64,
    ds::Float64,
    ws
)
    if status != STATUS_SUCCESS
        return status
    end

    dist_sq = (t_next - t_pred)^2
    @inbounds for j in 1:length(x_next)
        dist_sq += (x_next[j] - x_pred[j])^2
    end

    if dist_sq > (1.0 * ds)^2
        #    return STATUS_LARGE_DISTANCE
    end

    # lu determinant, but if a rcond heuristic is below 1e-5, it will return 0 to disable the check
    cur_det_sign = lu_det_sign_rcond_heur(ws.J_aug, ws.ipiv)
    if ws.det_sign[1] == 0.0 || cur_det_sign == 0.0
        ws.det_sign[1] = cur_det_sign
    elseif cur_det_sign != ws.det_sign[1]
        if (dist_sq > (0.5 * ds)^2 || ds > 1e-5)
            return STATUS_JUMP
        else
            ws.det_sign[1] = cur_det_sign
        end
    end

    return STATUS_SUCCESS
end


using Printf
function strat_format(xs)
    join([@sprintf("%.5e ", w) for w in xs])
end
function solve(
    utils::NTuple{N,Array{F,N}};
    stop_iters::Int=1000,
    stop_lambda::Float64=1e6,
    stop_eps::Float64=1e-6
) where {N,F<:Real}
    validate_game(utils)
    stop_t = log1p(stop_lambda)

    x = uniform_xprofile(utils)
    t = 0.0

    dx = zero(x)
    dt = 1.0
    ds = 0.01

    # Add tracking variables for curvature predictor
    dx_old = copy(dx)
    dt_old = dt
    ds_old = 0.0

    ws = make_hc_workspace(x, size(first(utils)))

    iteration = 0
    successes_in_row = 0
    regret = NaN
    stall = false
    while t <= stop_t && iteration <= stop_iters && !stall && t > -10.0
        update_predictor_jacobian!(x, t, dx, dt, utils, ws)
        _regret = max_deviation_incentive(ws.ubar, ws.pi)

        if _regret <= stop_eps
            regret = _regret
            break
        end

        dx, dt = predict_direction!(dx, ws)

        mu = splitviews(x, size(first(utils)) .- 1)
        for p in 1:N
            redlograt_to_prob!(ws.pi[p], mu[p], ws.active_actions[p], ws.refs[p])
        end

      #  println(strat_format(x), @sprintf(" %.5f", expm1(t)))

        # Prevent exponential runaway that causes Float64 Inf,
        # and avoid massively overshooting the stop condition.
        if t + ds * dt > stop_t + 2.0
            ds = (stop_t + 2.0 - t) / dt
        end

        while true
            x_pred, t_pred = predict_step_curvature!(ws.x_pred, x, t, dx, dt, dx_old, dt_old, ds, ds_old)
            st_cor, x_nxt, t_nxt = correct!(ws.x_nxt, x_pred, t_pred, dx, dt, utils, ws)
            st_val = validate_step!(st_cor, x_nxt, t_nxt, x_pred, t_pred, ds, ws)

            if st_val == STATUS_SUCCESS
                copyto!(x, x_nxt)
                t = t_nxt

                copyto!(dx_old, dx)
                dt_old = dt
                ds_old = ds

                # --- NEW: Dynamic pivot check block ---
                for p in 1:N
                    best_a = argmax(ws.pi[p])
                    if best_a != ws.refs[p]
                        idx = findfirst(==(best_a), ws.active_actions[p])
                        pivot_reference!(x, dx, dx_old, p, idx, ws)
                    end
                end
                # --------------------------------------

                break
            else
                ds /= 2.0
                successes_in_row = 0
                if ds <= 1e-12
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

    if isnan(regret)
        update_predictor_jacobian!(x, t, dx, dt, utils, ws)
        regret = max_deviation_incentive(ws.ubar, ws.pi)
    end

    return ws.pi, (; lambda=expm1(t), iteration, regret, stall)
end