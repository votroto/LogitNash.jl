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

    return (; pi, res, ubar, dudpi, J_aug, Fx, Ft, ipiv, rhs_aug, x_pred, x_nxt, det_sign)
end

function update_predictor_jacobian!(x::Vector{Float64}, t::Float64, dx::Vector{Float64}, dt::Float64, utils::NTuple{N}, ws) where {N}
    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
    unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

    lambda = expm1(t)

    jacobian_x!(ws.Fx, ws.pi, lambda, ws.dudpi, utils)
    jacobian_t!(ws.Ft, ws.ubar, mu, lambda)

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
        redlograt_to_prob!.(ws.pi, mu)

        unilateral_derivatives!(ws.dudpi, utils, ws.pi)
        unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

        lambda = expm1(t_nxt)

        residual!(ws.res, ws.ubar, mu, lambda)

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

        jacobian_x!(ws.Fx, ws.pi, lambda, ws.dudpi, utils)
        jacobian_t!(ws.Ft, ws.ubar, mu, lambda)

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

        println(strat_format(x)..., @sprintf(" %.5e", (t)))

        dx, dt = predict_direction!(dx, ws)

        # Prevent exponential runaway that causes Float64 Inf,
        # and avoid massively overshooting the stop condition.
        if t + ds * dt > stop_t + 2.0
            ds = (stop_t + 2.0 - t) / dt
        end

        while true
            # Pass curvature history to predictor
            x_pred, t_pred = predict_step_curvature!(ws.x_pred, x, t, dx, dt, dx_old, dt_old, ds, ds_old)
            st_cor, x_nxt, t_nxt = correct!(ws.x_nxt, x_pred, t_pred, dx, dt, utils, ws)
            st_val = validate_step!(st_cor, x_nxt, t_nxt, x_pred, t_pred, ds, ws)


   #     println(strat_format.(ws.pi)..., @sprintf(" %.5f", expm1(t)))

          #  @show st_val, st_cor
            if st_val == STATUS_SUCCESS
                copyto!(x, x_nxt)
                t = t_nxt

                copyto!(dx_old, dx)
                dt_old = dt
                ds_old = ds
                break
            else
                ds /= 2.0
                successes_in_row = 0
                if ds <= 1e-48
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






























function track_path!(
    x_start::Vector{Float64},
    lambda_start::Float64,
    dx_start::Vector{Float64},
    dlambda_start::Float64,
    utils::NTuple{N,AbstractArray{Float64,N}},
    ws;
    max_steps::Int=100
) where {N}
    x = copy(x_start)
    lambda = lambda_start
    dx = copy(dx_start)
    dlambda = dlambda_start

    # Reset determinant sign so the step validator doesn't reject
    # the first step based on history from a previous track.
    ws.det_sign[1] = 0.0

    ds = 0.01
    successes_in_row = 0



    # Add tracking variables for curvature predictor
    dx_old = copy(dx)
    dt_old = dlambda
    ds_old = 0.0

    for _ in 1:max_steps
        # Terminate if lambda goes completely out of bounds to avoid Inf
        if lambda > 20.0 || lambda < -20.0
            break
        end

        update_predictor_jacobian!(x, lambda, dx, dlambda, utils, ws)

        # Print current state. The spaces align perfectly for Gnuplot.
        println(strat_format.(x)..., @sprintf(" %.5e", (lambda)))

        # Update tangent direction
        dx, dlambda = predict_direction!(dx, ws)

        stall = false
        while true
            #xpred, lambda_pred = predict_step!(ws.x_pred, x, lambda, dx, dlambda, ds)
            xpred, lambda_pred = predict_step_curvature!(ws.x_pred, x, lambda, dx, dlambda, dx_old, dt_old, ds, ds_old)

            corr_status, x_nxt, lambda_nxt = correct!(ws.x_nxt, xpred, lambda_pred, dx, dlambda, utils, ws)
            val_status = validate_step!(corr_status, x_nxt, lambda_nxt, xpred, lambda_pred, ds, ws)

            if val_status == STATUS_SUCCESS
                copyto!(x, x_nxt)
                lambda = lambda_nxt

                   copyto!(dx_old, dx)
                dt_old = dlambda
                ds_old = ds

                break
            else
                ds /= 2.0
                successes_in_row = 0
                if ds <= 1e-10
                    stall = true
                    break
                end
            end
        end

        if stall
            break
        end

        successes_in_row += 1
        if successes_in_row >= 5
            successes_in_row = 0
            ds *= 2.0
        end
    end
end


function prob_to_redlograt(y::AbstractVector)
    x = similar(y, length(y)-1)
    for i in eachindex(x)
        x[i] = log(y[i] / y[end])
    end
    x
end

"""
    explore_manifold(utils; num_starts=50, steps_per_dir=100, lambda_range=(-5.0, 5.0), x_scale=2.0)

Randomly searches for points on the equilibrium manifold and tracks them in both directions.
Prints the visited points to stdout in a format ready for Gnuplot.
"""
function explore_manifold(
    utils::NTuple{N,AbstractArray{Float64,N}};
    num_starts::Int=200,
    steps_per_dir::Int=200,
    lambda_range::Tuple{Float64,Float64}=(1.0, 8.0),
    x_scale::Float64=2.0
) where {N}
    validate_game(utils)

    dims = size(first(utils))
    n = sum(dims[i] - 1 for i in 1:N)

    # Create workspace using a dummy template vector
    x_template = zeros(n)
    ws = make_hc_workspace(x_template, dims)

    # Pre-allocate variables for the random projection step
    dx_proj = zeros(n)
    dlambda_proj = 1.0

    for ii in 0:num_starts
        # 1. Random starting guess

        if ii==0
            x_guess = uniform_xprofile(utils)
            lambda_guess = 0.0

        elseif ii <= num_starts/2
            x_guess = vcat(map(d -> prob_to_redlograt(normalize(rand(d), 1)), dims)...) #randn(n) .* x_scale
            lambda_guess = 15.0 #rand(Float64) * (lambda_range[2] - lambda_range[1]) + lambda_range[1]
        else
            x_guess = vcat(map(d -> prob_to_redlograt(normalize(rand(d), 1)), dims)...) #randn(n) .* x_scale
            lambda_guess = rand(Float64) * (lambda_range[2] - lambda_range[1]) + lambda_range[1]
        end

        # 2. Project onto the manifold
        # Using dx=0, dlambda=1 means the corrector is constrained strictly
        # to the lambda_guess plane, giving Newton the freedom to solve for x.
        corr_status, x_root, lambda_root = correct!(
            ws.x_nxt, x_guess, lambda_guess,
            dx_proj, dlambda_proj, utils, ws;
            max_iters=200 # Allow more iterations for global convergence
        )

        if corr_status == STATUS_SUCCESS
            # 3. Establish initial tangent
            # We seed the Jacobian with our projection direction, and let
            # predict_direction! naturally find the null space.
            update_predictor_jacobian!(x_root, lambda_root, dx_proj, dlambda_proj, utils, ws)

            dx0 = copy(dx_proj)
            dx0, dlambda0 = predict_direction!(dx0, ws)
            ws = make_hc_workspace(x_template, dims)

            # 4. Track Forward
            track_path!(x_root, lambda_root, dx0, dlambda0, utils, ws; max_steps=steps_per_dir)

            # Gnuplot block separator (double blank line signifies a new dataset block)
            println("\n")
            ws = make_hc_workspace(x_template, dims)

            # 5. Track Backward
            # Flipping the signs on the tangent perfectly reverses the tracker
            track_path!(x_root, lambda_root, -dx0, -dlambda0, utils, ws; max_steps=steps_per_dir)

            println("\n")
        end
    end
end