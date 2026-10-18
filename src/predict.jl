function predict_tangent!(dx, dx_old, dt_old, t, pi, ws)
    lambda = expm1(t)

    system_jac_Fx!(ws.jac_aug, pi, lambda, ws.dudpi, ws.ubar, ws.refs)
    system_jac_Ft!(ws.jac_aug, ws.ubar, lambda, ws.refs)
    system_jac_dx_dt!(ws.jac_aug, dx_old, dt_old)
    system_rhs_0_1!(ws.rhs_aug)

    dir_x, dir_t, = bordered_solve!(ws.jac_aug, ws.rhs_aug, ws.ipiv)

    rnrm = 1 / sqrt(squared_norm(dir_x) + dir_t^2)
    @. dx = dir_x * rnrm
    dt = dir_t * rnrm

    return dx, dt
end

function predict_step!(
    x_pred::Vector{Float64},
    x::Vector{Float64},
    t::Float64,
    dx::Vector{Float64},
    dt::Float64,
    ds::Float64,
    dx_old::Vector{Float64},
    dt_old::Float64,
    ds_old::Float64
)
    quad::Bool = !isnan(ds_old)

    c_factor = (ds^2) / (2.0 * ds_old)
    @. x_pred = x + ds * dx + quad * (c_factor * (dx - dx_old))
    t_pred = t + ds * dt + quad * (c_factor * (dt - dt_old))

    return x_pred, t_pred
end