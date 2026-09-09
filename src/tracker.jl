using LinearAlgebra
using LinearAlgebra: BlasInt

const DECEL_FACTOR::Float64 = 2.236

function make_hc_workspace(x_template::Vector{Float64}, dims::NTuple{N}) where {N}
    n = length(x_template)

    pi = ntuple(i -> Vector{Float64}(undef, dims[i]), Val(N))
    res = Vector{Float64}(undef, n)
    ubar = ntuple(i -> zeros(dims[i]), Val(N))
    dudpi = ntuple(p -> ntuple(q -> fill(NaN, dims[p], dims[q]), Val(N)), Val(N))

    J_aug = zeros(n + 1, n + 1)
    Fx = view(J_aug, 1:n, 1:n)
    Ft = view(J_aug, 1:n, n + 1)
    ipiv = Vector{BlasInt}(undef, n + 1)

    rhs_aug = Vector{Float64}(undef, n + 1)

    x_pred = Vector{Float64}(undef, n)
    x_nxt = Vector{Float64}(undef, n)

    refs = Int[dims[i] for i in 1:N]

    return (; pi, res, ubar, dudpi, J_aug, Fx, Ft, ipiv, rhs_aug, x_pred, x_nxt, refs)
end

function _solve!(
    utils::NTuple{N,Array{R,N}},
    x::Vector{X}, t::X, dx::Vector{X}, dt::X, ds::X, ws;
    stop_iters::Int=1000,
    stop_lambda::Float64=1e6,
    stop_eps::Float64=1e-6
) where {N,R<:Real,X}
    stop_t = log1p(stop_lambda)

    dx_old = copy(dx)
    dt_old = dt
    ds_old = ds

    det_sign = 0.0

    iteration = 0
    successes_in_row = 0
    regret = NaN
    stall = false
    while t <= stop_t && iteration <= stop_iters && !stall && t > -10.0
        update_predictor_jacobian!(x, t, dx, dt, utils, ws)
        _regret = max_deviation_incentive(ws.ubar, ws.pi)

        @debug "Path" _id=:step pi=ws.pi lambda=expm1(t)

        if _regret <= stop_eps
            regret = _regret
            break
        end

        dx, dt = predict_direction!(dx, ws)

        if t + ds * dt > stop_t + 2.0
            ds = (stop_t + 2.0 - t) / dt
        end

        while true
            x_pred, t_pred = predict_step_quadratic!(ws.x_pred, x, t, dx, dt, ds, dx_old, dt_old, ds_old)
            st_cor, x_nxt, t_nxt = correct!(ws.x_nxt, x_pred, t_pred, dx, dt, utils, ws)
            st_val, det_nxt = validate_step(st_cor, x_nxt, t_nxt, x_pred, t_pred, det_sign, ds, ws)

            if st_val == STATUS_JUMP
                ds_mid = ds / DECEL_FACTOR
                x_mid, t_mid = predict_step_quadratic!(ws.x_pred, x, t, dx, dt, ds_mid, dx_old, dt_old, ds_old)
                st_val = validate_residual(x_mid, t_mid, utils, ws)
            end

            if st_val == STATUS_SUCCESS
                copyto!(x, x_nxt)
                t = t_nxt
                det_sign = det_nxt

                copyto!(dx_old, dx)
                dt_old = dt
                ds_old = ds

                pivot_references!(x, dx, dx_old, ws.pi, ws.refs)
                break
            else
                ds /= DECEL_FACTOR
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

    @debug "Path" _id=:end pi=ws.pi lambda=expm1(t)

    if isnan(regret)
        update_predictor_jacobian!(x, t, dx, dt, utils, ws)
        regret = max_deviation_incentive(ws.ubar, ws.pi)
    end

    return ws.pi, (; lambda=expm1(t), iteration, regret, stall)
end

function solve(utils::NTuple{N,Array{R,N}}; kwargs...) where {N,R<:Real}
    validate_game(utils)
    dims = size(first(utils))

    x = uniform_xprofile(dims)
    t = 0.0

    dx = zero(x)
    dt = 1.0
    ds = 0.01

    ws = make_hc_workspace(x, dims)

    _solve!(utils, x, t, dx, dt, ds, ws; kwargs...)
end
