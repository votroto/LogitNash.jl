@enum StepStatus begin
    STATUS_SUCCESS
    STATUS_MAX_ITERS
    STATUS_SINGULAR
    STATUS_LARGE_DISTANCE
    STATUS_JUMP
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
        mu, pi = extract_strategy_profiles!(ws.pi, x_nxt, ws.refs)
        lambda = expm1(t_nxt)

        unilateral_derivatives_cached!(ws.dudpi, utils, pi)
        unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, pi)
        residual!(ws.res, ws.ubar, mu, lambda, ws.refs)

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

        jacobian_x!(ws.Fx, pi, lambda, ws.dudpi, ws.ubar, ws.refs)
        jacobian_t!(ws.Ft, ws.ubar, lambda, ws.refs)
        jacobian_aug!(ws.J_aug, dx, dt)

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
        t_nxt += dt_step

        step_norm_2 = (step_norm_sq)
        val_norm_2 = dot(x_nxt, x_nxt) + t_nxt^2


        if !isfinite(t_nxt) || t_nxt > 50.0 || t_nxt < -20.0 || any(!isfinite, x_nxt)
            return STATUS_LARGE_DISTANCE, x_nxt, t_nxt
        end

        if step_norm_2 < rel_tol^2 * val_norm_2
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
        return STATUS_LARGE_DISTANCE
    end

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
