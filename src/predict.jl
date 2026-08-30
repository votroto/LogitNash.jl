function update_predictor_jacobian!(x::Vector{Float64}, t::Float64, dx::Vector{Float64}, dt::Float64, utils::NTuple{N}, ws) where {N}
    mu, pi = extract_strategy_profiles!(ws.pi, x, ws.refs)
    lambda = expm1(t)

    unilateral_derivatives!(ws.dudpi, utils, pi)
    unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, pi)

    jacobian_x!(ws.Fx, pi, lambda, ws.dudpi, ws.refs)
    jacobian_t!(ws.Ft, ws.ubar, mu, lambda, ws.refs)
    jacobian_aug!(ws.J_aug, dx, dt)

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

function predict_step_quadratic!(
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
    c_factor = (ds^2) / (2.0 * ds_old)
    @. x_pred = x + ds * dx + c_factor * (dx - dx_old)
    t_pred = t + ds * dt + c_factor * (dt - dt_old)

    return x_pred, t_pred
end