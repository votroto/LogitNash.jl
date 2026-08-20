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

function update_predictor_jacobian!(x::Vector{Float64}, t::Float64, dx::Vector{Float64}, dt::Float64, utils::NTuple{N}, ws) where {N}
    n = length(x)
    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
    unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

    jacobian_x!(ws.Fx, ws.pi, t, ws.dudpi, utils)
    jacobian_t!(ws.Ft, ws.ubar, mu, utils)

    @inbounds for j in 1:n
        ws.J_aug[end, j] = dx[j]
    end
    ws.J_aug[end, end] = dt

    n = length(dx)
    fill!(ws.rhs_aug, 0.0)
    ws.rhs_aug[end] = 1.0

    fast_lu!(ws.J_aug, ws.ipiv)

    nothing
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

function predict_step!(xpred::Vector{Float64}, x::Vector{Float64}, t::Float64, dx::Vector{Float64}, dt::Float64, ds::Float64)
    @. xpred = x + ds * dx
    tpred = t + ds * dt
    return xpred, tpred
end

function predict_step_parabolic!(
    xpred::Vector{Float64}, x::Vector{Float64}, t::Float64,
    dx::Vector{Float64}, dt::Float64, ds::Float64,
    dx_last::Vector{Float64}, dt_last::Float64, ds_last::Float64, has_last::Bool
)
    if has_last
        # Second order Hermite/Secant curve
        c = 0.5 * (ds^2 / ds_last)
        @. xpred = x + ds * dx + c * (dx - dx_last)
        tpred = t + ds * dt + c * (dt - dt_last)
    else
        # Fallback to Euler for the first step
        @. xpred = x + ds * dx
        tpred = t + ds * dt
    end
    return xpred, tpred
end


function predict_init!(x::Vector{Float64}, t::Float64, utils::NTuple{N}, ws) where {N}
    n = length(x)
    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
    unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

    jacobian_x!(ws.Fx, ws.pi, t, ws.dudpi, utils)
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

function correct!(
    x_nxt::Vector{Float64},
    xpred::Vector{Float64},
    tpred::Float64,
    dx::Vector{Float64},
    dt::Float64,
    utils::NTuple{N},
    ws;
    max_iters::Int=4,
    abs_tol::Float64=1e-6,
    rel_tol::Float64=1e-12
) where {N}
    copyto!(x_nxt, xpred)
    t_out = tpred
    n = length(x_nxt)

    prev_step_norm = Inf

    for i in 0:max_iters
        mu = splitviews(x_nxt, size(first(utils)) .- 1)
        redlograt_to_prob!.(ws.pi, mu)

        unilateral_derivatives!(ws.dudpi, utils, ws.pi)
        unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

        residual!(ws.res, mu, ws.ubar, x_nxt, t_out, utils)

        @. ws.x_diff = x_nxt - xpred
        r_con = dot(ws.x_diff, dx) + (t_out - tpred) * dt

        jacobian_x!(ws.Fx, ws.pi, t_out, ws.dudpi, utils)
        jacobian_t!(ws.Ft, ws.ubar, mu, utils)

        scale_factor = max(1.0, t_out)

        @inbounds for j in 1:n
            ws.J_aug[end, j] = dx[j] * scale_factor
        end
        ws.J_aug[end, end] = dt * scale_factor

        @inbounds for j in 1:n
            ws.rhs_aug[j] = -ws.res[j]
        end
        ws.rhs_aug[end] = -r_con * scale_factor

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

        if i >= 1 && step_norm > 0.5 * prev_step_norm
            return STATUS_LARGE_DISTANCE, i, x_nxt, t_out
        end

        prev_step_norm = step_norm

        if step_norm < 1e-14 * val_norm
            return STATUS_SUCCESS, i, x_nxt, t_out
        end

        @. x_nxt += ws.dx_step
        t_out += dt_step

        if step_norm < rel_tol * val_norm && i >= 1
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

    dist_sq = (t_new - tpred)^2
    @inbounds for j in 1:length(x_nxt)
        dist_sq += (x_nxt[j] - xpred[j])^2
    end

    if dist_sq > max((2.0 * ds)^2, 1e-10)
        return STATUS_LARGE_DISTANCE
    end

    cur_det_sign = lu_det_sign(ws.J_aug, ws.ipiv)
    if ws.det_sign[1] != 0.0 && cur_det_sign != ws.det_sign[1]
        if dist_sq > max((0.5 * ds)^2, 2.5e-11) || ds > 1e-5
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
Compute an epsilon Nash equilibrium of an N-player game by tracing a logit equilibrium path from t=0 to infinity.
"""
function nash(
    utils::NTuple{N,AbstractArray{Float64,N}};
    stop_iters::Int=1000,
    stop_t::Float64=1e6,
    stop_eps::Float64=1e-6
) where {N}
    validate_game(utils)

    x = uniform_xprofile(utils)
    x .+= rand(length(x))*1e-3
    t = 0.0
    dx = zero(x)
    dt = 1.0
    ds = 0.012

    dx_last=copy(dx)
    dt_last = dt
    ds_last=ds
    has_last=false

    ws = make_hc_workspace(x, size(first(utils)))
    predict_init!(x, t, utils, ws)

    iteration = 0
    successes_in_row = 0
    regret = NaN
    stall = false

    while t <= stop_t && iteration <= stop_iters && !stall && t > -10
        update_predictor_jacobian!(x, t, dx, dt, utils, ws)

        dx, dt = predict_direction!(dx, dt, ws)
        regret = max_deviation_incentive(ws.ubar, ws.pi)
      #  println(strat_format.(ws.pi)..., "$(ws.det_sign[1]) $(round(t;digits=3)) $(round(regret;digits=5)) ", strat_format(dx), " $(round(dt;digits=4))", "\t", copysign(round(prod(diag(ws.J_aug));digits=3), ws.det_sign[1]))

        if regret <= stop_eps
            break
        end

        while true
            #xpred, tpred = predict_step!(ws.xpred, x, t, dx, dt, ds)
            xpred, tpred = predict_step_parabolic!(ws.xpred, x, t, dx, dt, ds, dx_last, dt_last, ds_last, has_last)

            corr_status, iters, x_nxt, t_nxt = correct!(ws.x_nxt, xpred, tpred, dx, dt, utils, ws)
            val_status = validate_step!(corr_status, iters, x_nxt, t_nxt, xpred, tpred, ds, ws)

# --- BEGIN DEBUG INJECTION ---
        if t > 1700.0 && ds < 1e-4
            println("\n--- DEBUG: TRACKING STALL ---")
            println("Current t    = ", t)
            println("Current ds   = ", ds)
            println("Tangent dt   = ", dt)
            println("Norm(dx)     = ", norm(dx))
            println("Corr status  = ", corr_status, " (iters: ", iters, ")")
            println("Val status   = ", val_status)

            # Reconstruct the augmented Jacobian BEFORE LU destroys it to check conditioning
            mu_pred = splitviews(xpred, size(first(utils)) .- 1)
            # Assuming you have a temporary pi workspace or can just overwrite ws.pi safely here since step failed
            redlograt_to_prob!.(ws.pi, mu_pred)
            unilateral_derivatives!(ws.dudpi, utils, ws.pi)
            unilateral_deviations_from_derivatives!(ws.ubar, ws.dudpi, ws.pi)

            residual!(ws.res, mu_pred, ws.ubar, xpred, tpred, utils)
            println("Pred Res Norm = ", norm(ws.res))

            jacobian_x!(ws.Fx, ws.pi, tpred, ws.dudpi, utils)
            jacobian_t!(ws.Ft, ws.ubar, mu_pred, utils)

            # Rebuild the exact matrix passed to LAPACK in correct!
            J_test = copy(ws.J_aug)
            scale_fac = max(1.0, tpred)
            for j in 1:length(x)
                J_test[end, j] = dx[j] * scale_fac
            end
            J_test[end, end] = dt * scale_fac

            println("Cond(J_aug)   = ", cond(J_test))
            println("-----------------------------")
        end
        # --- END DEBUG INJECTION ---

            if val_status == STATUS_SUCCESS
                copyto!(x, x_nxt)
                t = t_nxt

                copyto!(dx_last, dx)
                dt_last = dt
                ds_last = ds
                has_last = true
                break
            else
                if ds <= 1e-6
                    x .+= rand(length(x))*1e-3
                end
                ds /= 2.0
                successes_in_row = 0
                if ds <= 1e-8
                    stall = true
                    break
                end
            end
        end

        successes_in_row += 1
        if successes_in_row >= 5
            successes_in_row = 0
            ds *= 1.9
        end
        iteration += 1
    end

    return ws.pi, (; t, iteration, regret, stall)
end