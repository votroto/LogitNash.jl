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
    sum(maximum(ubar[p]) - dot(ubar[p], pi[p]) for p in 1:N)
end

function make_hc_workspace(x_template::Vector{Float64}, dims::NTuple{N}) where {N}
    T = eltype(x_template)
    n = length(x_template)
    rsize = sum(dims[i] - 1 for i in 1:N)

    pi = ntuple(i -> Vector{T}(undef, dims[i]), Val(N))
    res = Vector{T}(undef, rsize)
    ubar = ntuple(i -> zeros(T, dims[i]), Val(N))
    dudpi = ntuple(p -> ntuple(q -> zeros(T, dims[p], dims[q]), Val(N)), Val(N))

    J_aug = zeros(T, n + 1, n + 1)
    Fx = view(J_aug, 1:n, 1:n)
    Ft = view(J_aug, 1:n, n + 1)
    ipiv = Vector{BlasInt}(undef, n + 1)

    rhs_aug = Vector{T}(undef, n + 1)

    xpred = Vector{T}(undef, n)
    x_diff = Vector{T}(undef, n)
    dx_step = Vector{T}(undef, n)
    x_nxt = Vector{Float64}(undef, n)

    det_sign = Float64[1.0]

    return (; pi, res, ubar, dudpi, J_aug, Fx, Ft, ipiv, rhs_aug, xpred, x_diff, dx_step, x_nxt, det_sign)
end

function lu_det_sign_and_rcond(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    s = 1.0
    min_d = Inf
    max_d = 0.0
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
    # Add a tiny epsilon to prevent divide-by-zero
    rcond = min_d / (max_d + 1e-18)
    return s, rcond
end

function fast_lu!(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    A, ipiv, info = LinearAlgebra.LAPACK.getrf!(A, ipiv)
    return info
end

function update_predictor_jacobian!(x::Vector{Float64}, lambda::Float64, dx::Vector{Float64}, dlambda::Float64, utils::NTuple{N}, ws) where {N}
    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
    unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

    t_val = expm1(lambda)

    jacobian_x!(ws.Fx, ws.pi, t_val, ws.dudpi, utils)
    jacobian_t!(ws.Ft, ws.ubar, mu, utils, t_val)

    @inbounds for j in eachindex(dx)
        ws.J_aug[end, j] = dx[j]
    end
    ws.J_aug[end, end] = dlambda

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
    dlambda = ws.rhs_aug[end] * norm_factor

    return dx, dlambda
end

function predict_step!(xpred::Vector{Float64}, x::Vector{Float64}, lambda::Float64, dx::Vector{Float64}, dlambda::Float64, ds::Float64)
    @. xpred = x + ds * dx
    lambda_pred = lambda + ds * dlambda
    return xpred, lambda_pred
end

function correct!(
    x_nxt::Vector{Float64},
    xpred::Vector{Float64},
    lambda_pred::Float64,
    dx::Vector{Float64},
    dlambda::Float64,
    utils::NTuple{N},
    ws;
    max_iters::Int=3,
    abs_tol::Float64=1e-9,
    rel_tol::Float64=1e-11
) where {N}
    copyto!(x_nxt, xpred)
    lambda_out = lambda_pred

    for i in 0:max_iters
        mu = splitviews(x_nxt, size(first(utils)) .- 1)
        redlograt_to_prob!.(ws.pi, mu)

        unilateral_derivatives!(ws.dudpi, utils, ws.pi)
        unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

        t_val = expm1(lambda_out)

        residual!(ws.res, mu, ws.ubar, x_nxt, t_val, utils)

        @. ws.x_diff = x_nxt - xpred
        r_con = dot(ws.x_diff, dx) + (lambda_out - lambda_pred) * dlambda

        if dot(ws.res, ws.res) + r_con^2 < abs_tol^2
            return STATUS_SUCCESS, x_nxt, lambda_out
        end

        if i == max_iters
            return STATUS_MAX_ITERS, x_nxt, lambda_out
        end

        jacobian_x!(ws.Fx, ws.pi, t_val, ws.dudpi, utils)
        jacobian_t!(ws.Ft, ws.ubar, mu, utils, t_val)

        @inbounds for j in eachindex(dx)
            ws.J_aug[end, j] = dx[j]
        end
        ws.J_aug[end, end] = dlambda

        @inbounds for j in eachindex(ws.res)
            ws.rhs_aug[j] = -ws.res[j]
        end
        ws.rhs_aug[end] = -r_con


        info = fast_lu!(ws.J_aug, ws.ipiv)
        if info > 0
            return STATUS_SINGULAR, x_nxt, lambda_out
        end

        LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

        dlambda_step = ws.rhs_aug[end]
        step_norm_sq = dlambda_step^2

        @inbounds for j in eachindex(ws.dx_step)
            ws.dx_step[j] = ws.rhs_aug[j]
            step_norm_sq += ws.dx_step[j]^2
        end

        step_norm = sqrt(step_norm_sq)
        val_norm = sqrt(dot(x_nxt, x_nxt) + lambda_out^2)

        @. x_nxt += ws.dx_step
        lambda_out += dlambda_step

        # Catch wildly diverging Newton steps before they corrupt the Jacobian with Infs.
        if !isfinite(lambda_out) || lambda_out > 50.0 || lambda_out < -20.0 || any(!isfinite, x_nxt)
            return STATUS_LARGE_DISTANCE, x_nxt, lambda_out
        end

        if step_norm < rel_tol * val_norm
            return STATUS_SUCCESS, x_nxt, lambda_out
        end
    end

    return STATUS_MAX_ITERS, x_nxt, lambda_out
end

function validate_step!(
    status::StepStatus,
    x_nxt::Vector{Float64},
    lambda_new::Float64,
    xpred::Vector{Float64},
    lambda_pred::Float64,
    ds::Float64,
    ws
)
    if status != STATUS_SUCCESS
        return status
    end

    dist_sq = (lambda_new - lambda_pred)^2
    @inbounds for j in 1:length(x_nxt)
        dist_sq += (x_nxt[j] - xpred[j])^2
    end

    if dist_sq > (1.0 * ds)^2
        return STATUS_LARGE_DISTANCE
    end

    cur_det_sign, rcond = lu_det_sign_and_rcond(ws.J_aug, ws.ipiv)
    if rcond >= 1e-5
        if ws.det_sign[1] == 0.0
            ws.det_sign[1] = cur_det_sign
        elseif cur_det_sign != ws.det_sign[1]
            if (dist_sq > (0.5 * ds)^2 || ds > 1e-5)
                return STATUS_JUMP
            else
                ws.det_sign[1] = cur_det_sign
            end
        end
    else
        ws.det_sign[1] = 0.0
    end

    return STATUS_SUCCESS
end


using Printf
function strat_format(xs)
    join([@sprintf("%.5f ", w) for w in xs])
end


"""
    nash(utils::NTuple{N,AbstractArray{Float64,N}}; stop_iters::Int=1000, stop_t::Float64=1e6, stop_eps::Float64=1e-6) where {N}
"""
function nash(
    utils::NTuple{N,AbstractArray{F,N}};
    stop_iters::Int=1000,
    stop_t::Float64=1e6,
    stop_eps::Float64=1e-6
) where {F,N}
    validate_game(utils)

    x = uniform_xprofile(utils)

    lambda = 0.0
    stop_lambda = log1p(stop_t)

    dx = zero(x)
    dlambda = 1.0
    ds = 0.01

    ws = make_hc_workspace(x, size(first(utils)))

    iteration = 0
    successes_in_row = 0
    regret = NaN
    stall = false
    while lambda <= stop_lambda && iteration <= stop_iters && !stall #&& lambda > -10.0
        update_predictor_jacobian!(x, lambda, dx, dlambda, utils, ws)

        regret = max_deviation_incentive(ws.ubar, ws.pi)

        #  println(strat_format.(ws.pi)..., @sprintf(" %.5f", expm1(lambda)))

        if regret <= stop_eps
            break
        end

        dx, dlambda = predict_direction!(dx, ws)

        # Prevent exponential runaway that causes Float64 Inf,
        # and avoid massively overshooting the stop condition.
        if lambda + ds * dlambda > stop_lambda + 2.0
            ds = (stop_lambda + 2.0 - lambda) / dlambda
        end

        while true
            xpred, lambda_pred = predict_step!(ws.xpred, x, lambda, dx, dlambda, ds)

            corr_status, x_nxt, lambda_nxt = correct!(ws.x_nxt, xpred, lambda_pred, dx, dlambda, utils, ws)
            val_status = validate_step!(corr_status, x_nxt, lambda_nxt, xpred, lambda_pred, ds, ws)

            if val_status == STATUS_SUCCESS
                copyto!(x, x_nxt)
                lambda = lambda_nxt
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

    update_predictor_jacobian!(x, lambda, dx, dlambda, utils, ws)
    regret = max_deviation_incentive(ws.ubar, ws.pi)

    return ws.pi, (; t=expm1(lambda), iteration, regret, stall)
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

    for _ in 1:max_steps
        # Terminate if lambda goes completely out of bounds to avoid Inf
        if lambda > 20.0 || lambda < -20.0
            break
        end

        update_predictor_jacobian!(x, lambda, dx, dlambda, utils, ws)

        # Print current state. The spaces align perfectly for Gnuplot.
        println(strat_format.(ws.pi)..., @sprintf("%.5f", expm1(lambda)))

        # Update tangent direction
        dx, dlambda = predict_direction!(dx, ws)

        stall = false
        while true
            xpred, lambda_pred = predict_step!(ws.xpred, x, lambda, dx, dlambda, ds)

            corr_status, x_nxt, lambda_nxt = correct!(ws.x_nxt, xpred, lambda_pred, dx, dlambda, utils, ws)
            val_status = validate_step!(corr_status, x_nxt, lambda_nxt, xpred, lambda_pred, ds, ws)

            if val_status == STATUS_SUCCESS
                copyto!(x, x_nxt)
                lambda = lambda_nxt
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
    lambda_range::Tuple{Float64,Float64}=(0.0, 8.0),
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