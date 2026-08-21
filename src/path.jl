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


function _amax_deviation_incentive(
    deviations::NTuple{N,Vector{Float64}},
    xs::NTuple{N,Vector{Float64}}
) where N
    actuals = dot.(deviations, xs)
    bests = maximum.(deviations)

    maximum(bests[p] - actuals[p] for p in 1:N)
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

function update_predictor_jacobian!(x::Vector{Float64}, lambda::Float64, dx::Vector{Float64}, dlambda::Float64, utils::NTuple{N}, ws) where {N}
    n = length(x)
    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
   # for p in 1:N
   #        fill!(ws.ubar[p],0.0)
   #     end
    unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

    t_val = exp(lambda) - 1.0

    jacobian_x!(ws.Fx, ws.pi, t_val, ws.dudpi, utils)
    jacobian_t!(ws.Ft, ws.ubar, mu, utils)

    # Scale the parameter column from Ft to Fλ
    fac = exp(lambda)
    @inbounds for i in 1:n
        ws.Ft[i] *= fac
    end

    @inbounds for j in 1:n
        ws.J_aug[end, j] = dx[j]
    end
    ws.J_aug[end, end] = dlambda

    fill!(ws.rhs_aug, 0.0)
    ws.rhs_aug[end] = 1.0

    fast_lu!(ws.J_aug, ws.ipiv)

    nothing
end

function predict_direction!(dx::Vector{Float64}, dlambda::Float64, ws)
    n = length(dx)
    fill!(ws.rhs_aug, 0.0)
    ws.rhs_aug[end] = 1.0

    LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

    norm_factor = 1.0 / sqrt(dot(ws.rhs_aug, ws.rhs_aug))
    @inbounds for i in 1:n
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

function predict_init!(x::Vector{Float64}, lambda::Float64, utils::NTuple{N}, ws) where {N}
    n = length(x)
    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
    unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

    t_val = exp(lambda) - 1.0

    jacobian_x!(ws.Fx, ws.pi, t_val, ws.dudpi, utils)
    jacobian_t!(ws.Ft, ws.ubar, mu, utils)

    # Scale the parameter column from Ft to Fλ
    fac = exp(lambda)
    @inbounds for i in 1:n
        ws.Ft[i] *= fac
    end

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


function correct!(
    x_nxt::Vector{Float64},
    xpred::Vector{Float64},
    lambda_pred::Float64,
    dx::Vector{Float64},
    dlambda::Float64,
    utils::NTuple{N},
    ws;
    max_iters::Int=5,
    abs_tol::Float64=1e-9,
    rel_tol::Float64=1e-12
) where {N}
    copyto!(x_nxt, xpred)
    lambda_out = lambda_pred
    n = length(x_nxt)

    for i in 0:max_iters
        mu = splitviews(x_nxt, size(first(utils)) .- 1)
        redlograt_to_prob!.(ws.pi, mu)

        unilateral_derivatives!(ws.dudpi, utils, ws.pi)
        unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

        t_val = exp(lambda_out) - 1.0

        residual!(ws.res, mu, ws.ubar, x_nxt, t_val, utils)

        @. ws.x_diff = x_nxt - xpred
        r_con = dot(ws.x_diff, dx) + (lambda_out - lambda_pred) * dlambda

        if dot(ws.res, ws.res) + r_con^2 < abs_tol^2 && i >= 1
            return STATUS_SUCCESS, i, x_nxt, lambda_out
        end

        if i == max_iters
            return STATUS_MAX_ITERS, max_iters, x_nxt, lambda_out
        end

        jacobian_x!(ws.Fx, ws.pi, t_val, ws.dudpi, utils)
        jacobian_t!(ws.Ft, ws.ubar, mu, utils)

        # Scale the parameter column from Ft to Fλ
        fac = exp(lambda_out)
        @inbounds for j in 1:n
            ws.Ft[j] *= fac
        end

        @inbounds for j in 1:n
            ws.J_aug[end, j] = dx[j]
        end
        ws.J_aug[end, end] = dlambda

        # Removed display(ws.J_aug) to prevent console spam, but you can add it back to verify the O(t) scaling
        # display(ws.J_aug)
        # println()

        @inbounds for j in 1:n
            ws.rhs_aug[j] = -ws.res[j]
        end
        ws.rhs_aug[end] = -r_con


        info = fast_lu!(ws.J_aug, ws.ipiv)
         if info > 0
            return STATUS_SINGULAR, i, x_nxt, lambda_out
         end

        LinearAlgebra.LAPACK.getrs!('N', ws.J_aug, ws.ipiv, ws.rhs_aug)

        dlambda_step = ws.rhs_aug[end]
        step_norm_sq = dlambda_step^2

        @inbounds for j in 1:n
            ws.dx_step[j] = ws.rhs_aug[j]
            step_norm_sq += ws.dx_step[j]^2
        end

        step_norm = sqrt(step_norm_sq)
        # val_norm now uses lambda, making it a much more balanced relative check than using raw t
        val_norm = sqrt(dot(x_nxt, x_nxt) + lambda_out^2)

        @. x_nxt += ws.dx_step
        lambda_out += dlambda_step

        # --- ADD THIS SAFEGUARD ---
        # Catch wildly diverging Newton steps before they corrupt the Jacobian with Infs.
        # This aborts the correction and allows the outer loop to safely halve the step size.
        if !isfinite(lambda_out) || lambda_out > 50.0 || lambda_out < -20.0 || any(!isfinite, x_nxt)
            return STATUS_LARGE_DISTANCE, i, x_nxt, lambda_out
        end

        if step_norm < rel_tol * val_norm  && i >= 1
            return STATUS_SUCCESS, i + 1, x_nxt, lambda_out
        end
    end

    return STATUS_MAX_ITERS, max_iters, x_nxt, lambda_out
end

function validate_step!(
    status::StepStatus,
    iters::Int,
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

    # Distance check now operates in log-scale lambda space!
    dist_sq = (lambda_new - lambda_pred)^2
    @inbounds for j in 1:length(x_nxt)
        dist_sq += (x_nxt[j] - xpred[j])^2
    end

    if dist_sq > (2.0 * ds)^2
        return STATUS_LARGE_DISTANCE
    end

    cur_det_sign = lu_det_sign(ws.J_aug, ws.ipiv)
    if ws.det_sign[1] != 0.0 && cur_det_sign != ws.det_sign[1]
        if dist_sq > (0.5 * ds)^2 || ds > 1e-5
            return STATUS_JUMP
        else
            ws.det_sign[1] = cur_det_sign
        end
    end

    return STATUS_SUCCESS
end


#using Printf
#function strat_format(xs) join([@sprintf("%.6f ", w) for w in xs]) end


"""
    nash(utils::NTuple{N,AbstractArray{Float64,N}}; stop_iters::Int=1000, stop_t::Float64=1e6, stop_eps::Float64=1e-6) where {N}
"""
function nash(
    utils::NTuple{N,AbstractArray{Float64,N}};
    stop_iters::Int=1000,
    stop_t::Float64=1e6,
    stop_eps::Float64=1e-6
) where {N}
    validate_game(utils)

    x = uniform_xprofile(utils)

    # Initialize in log space: ln(0 + 1) = 0
    lambda = 0.0
    stop_lambda = log(stop_t + 1.0)

    dx = zero(x)
    dlambda = 1.0
    ds = 0.01

    ws = make_hc_workspace(x, size(first(utils)))
    predict_init!(x, lambda, utils, ws)

    iteration = 0
    successes_in_row = 0
    regret = NaN
    stall = false

    while lambda <= stop_lambda && iteration <= stop_iters && !stall && lambda > -10.0
        update_predictor_jacobian!(x, lambda, dx, dlambda, utils, ws)

    #unilateral_deviations_simple!(ws.ubar, utils, ws.pi)

        regret = max_deviation_incentive(ws.ubar, ws.pi)

        t_val = exp(lambda) - 1.0
       # println(strat_format.(ws.pi)..., "$(ws.det_sign[1]) $(round(t_val;digits=3)) $(round(ds;digits=5))")

        if regret <= stop_eps
            break
        end

        dx, dlambda = predict_direction!(dx, dlambda, ws)

        while true
            # --- ADD THIS LIMITER ---
            # Prevent exponential runaway that causes Float64 Inf,
            # and avoid massively overshooting the stop condition.
            if dlambda > 0 && lambda + ds * dlambda > stop_lambda + 2.0
                ds = (stop_lambda + 2.0 - lambda) / dlambda
            end
            # ------------------------

            xpred, lambda_pred = predict_step!(ws.xpred, x, lambda, dx, dlambda, ds)

            corr_status, iters, x_nxt, lambda_nxt = correct!(ws.x_nxt, xpred, lambda_pred, dx, dlambda, utils, ws)
            val_status = validate_step!(corr_status, iters, x_nxt, lambda_nxt, xpred, lambda_pred, ds, ws)

            if val_status == STATUS_SUCCESS
                copyto!(x, x_nxt)
                lambda = lambda_nxt
                break
            else
                if ds <= 1e-6
                    x .+= rand(length(x))*1e-3
                end
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
  #  @show ws.ubar

    # Return raw t out to the user for compatibility
    return ws.pi, (; t = exp(lambda) - 1.0, iteration, regret, stall)
end