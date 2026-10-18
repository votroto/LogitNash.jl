@enum StepStatus begin
    STATUS_SUCCESS
    STATUS_MAX_ITERS
    STATUS_SINGULAR
    STATUS_LARGE_DISTANCE
    STATUS_JUMP
end

function correct!(
    x_next::Vector{Float64},
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
    copyto!(x_next, x_pred)
    t_next = t_pred

    for i in 0:max_iters
        mu, pi = extract_strategy_profiles!(ws.pi, x_next, ws.refs)
        lambda = expm1(t_next)

        update_kernels!(ws.dudpi, ws.ubar, pi, x_next, ws.refs, utils)
        residual!(ws.res, ws.ubar, mu, lambda, ws.refs)
        arc = pseudo_arclength(x_next, t_next, x_pred, t_pred, dx, dt)

        if squared_norm(ws.res) + arc^2 < abs_tol^2
            return STATUS_SUCCESS, x_next, t_next
        end

        if i == max_iters
            break
        end

        system_jac_Fx!(ws.jac_aug, pi, lambda, ws.dudpi, ws.ubar, ws.refs)
        system_jac_Ft!(ws.jac_aug, ws.ubar, lambda, ws.refs)
        system_jac_dx_dt!(ws.jac_aug, dx, dt)
        system_rhs_F_g!(ws.rhs_aug, ws.res, arc)

        x_step, t_step, info = bordered_solve!(ws.jac_aug, ws.rhs_aug, ws.ipiv)
        if info > 0
            return STATUS_SINGULAR, x_next, t_next
        end

        @. x_next += x_step
        t_next += t_step

        val_squared_norm = squared_norm(x_next) + t_next^2
        step_squared_norm = squared_norm(x_step) + t_step^2

        if any(!isfinite, x_next) || !(-10 <= t_next && t_next <= 50.0)
            return STATUS_LARGE_DISTANCE, x_next, t_next
        end

        # Further progress in f64 likely meaningless.
        if step_squared_norm < rel_tol^2 * val_squared_norm
            return STATUS_SUCCESS, x_next, t_next
        end
    end

    return STATUS_MAX_ITERS, x_next, t_next
end

function validate_step(
    status::StepStatus,
    x_next::Vector{Float64},
    t_next::Float64,
    x_pred::Vector{Float64},
    t_pred::Float64,
    det_sign::Float64,
    ds::Float64,
    ws
)
    if status != STATUS_SUCCESS
        return status, 0.0
    end

    squared_tangent_distance = (t_next - t_pred)^2
    @inbounds @simd for j in 1:length(x_next)
        squared_tangent_distance += (x_next[j] - x_pred[j])^2
    end

    if squared_tangent_distance > (0.5 * ds)^2
        return STATUS_LARGE_DISTANCE, 0.0
    end

    cur_det_sign = lu_det_sign_rcond_heur(ws.jac_aug, ws.ipiv)
    if det_sign != 0.0 && cur_det_sign != 0.0 && cur_det_sign != det_sign
        return STATUS_JUMP, cur_det_sign
    end

    return STATUS_SUCCESS, cur_det_sign
end

function validate_residual(x, t, utils, ws; abs_tol=1e-3)
    mu, pi = extract_strategy_profiles!(ws.pi, x, ws.refs)
    lambda = expm1(t)

    update_kernels!(ws.dudpi, ws.ubar, pi, x, ws.refs, utils)
    residual!(ws.res, ws.ubar, mu, lambda, ws.refs)

    if squared_norm(ws.res) <= abs_tol^2
        return STATUS_SUCCESS
    else
        return STATUS_JUMP
    end
end