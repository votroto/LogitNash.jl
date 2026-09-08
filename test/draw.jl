
using LogitNash
using LinearAlgebra
using Printf

function track_path!(x_start, t_start, dx_start, dt_start, utils, ws; stop_iters=100, stop_t=14.0)

    x = copy(x_start)
    t = t_start
    dx = copy(dx_start)
    dt = dt_start
    ds = 0.01

    dx_old = copy(dx)
    dt_old = dt
    ds_old = ds

    iteration = 0
    successes_in_row = 0
    regret = NaN
    stall = false
    while t <= stop_t && iteration <= stop_iters && !stall && t > -10.0
        _mu, pi = LogitNash.extract_strategy_profiles!(ws.pi, x, ws.refs)

        for pii in pi
            for xi in pii
                @printf "%.5e " xi
            end
        end
        @printf "%.5e \n" t

        LogitNash.update_predictor_jacobian!(x, t, dx, dt, utils, ws)

        dx, dt = LogitNash.predict_direction!(dx, ws)

        if t + ds * dt > stop_t + 2.0
            ds = (stop_t + 2.0 - t) / dt
        end

        while true
            x_pred, t_pred = LogitNash.predict_step_quadratic!(ws.x_pred, x, t, dx, dt, ds, dx_old, dt_old, ds_old)
            st_cor, x_nxt, t_nxt = LogitNash.correct!(ws.x_nxt, x_pred, t_pred, dx, dt, utils, ws)
            st_val = LogitNash.validate_step!(st_cor, x_nxt, t_nxt, x_pred, t_pred, ds, ws)

            if st_val == LogitNash.STATUS_SUCCESS
                copyto!(x, x_nxt)
                t = t_nxt

                copyto!(dx_old, dx)
                dt_old = dt
                ds_old = ds

                LogitNash.pivot_references!(x, dx, dx_old, ws.pi, ws.refs)
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
    utils::NTuple{N};
    num_starts::Int=200,
    steps_per_dir::Int=200,
    lambda_range::Tuple{Float64,Float64}=(1.0, 8.0),
) where {N}
    dims = size(first(utils))
    n = sum(dims[i] - 1 for i in 1:N)

    # Create workspace using a dummy template vector
    x_template = zeros(n)
    ws = LogitNash.make_hc_workspace(x_template, dims)

    # Pre-allocate variables for the random projection step
    dx_proj = zeros(n)
    dlambda_proj = 1.0

    for ii in 0:num_starts
        # 1. Random starting guess

        if ii==0
            x_guess = LogitNash.uniform_xprofile(utils)
            lambda_guess = 0.0
        elseif ii <= num_starts/2
            x_guess = vcat(map(d -> prob_to_redlograt(normalize(rand(d), 1)), dims)...)
            lambda_guess = 15.0
        else
            x_guess = vcat(map(d -> prob_to_redlograt(normalize(rand(d), 1)), dims)...)
            lambda_guess = rand(Float64) * (lambda_range[2] - lambda_range[1]) + lambda_range[1]
        end

        # 2. Project onto the manifold
        # Using dx=0, dlambda=1 means the corrector is constrained strictly
        # to the lambda_guess plane, giving Newton the freedom to solve for x.
        corr_status, x_root, lambda_root = LogitNash.correct!(
            ws.x_nxt, x_guess, lambda_guess,
            dx_proj, dlambda_proj, utils, ws;
            max_iters=200 # Allow more iterations for global convergence
        )

        if corr_status == LogitNash.STATUS_SUCCESS
            # 3. Establish initial tangent
            # We seed the Jacobian with our projection direction, and let
            # predict_direction! naturally find the null space.
            LogitNash.update_predictor_jacobian!(x_root, lambda_root, dx_proj, dlambda_proj, utils, ws)

            dx0 = copy(dx_proj)
            dx0, dlambda0 = LogitNash.predict_direction!(dx0, ws)
            ws = LogitNash.make_hc_workspace(x_template, dims)

            # 4. Track Forward
            track_path!(x_root, lambda_root, dx0, dlambda0, utils, ws; stop_iters=steps_per_dir)

            # Gnuplot block separator (double blank line signifies a new dataset block)
            println("\n")
     #       ws = LogitNash.make_hc_workspace(x_template, dims)
#
     #       # 5. Track Backward
     #       # Flipping the signs on the tangent perfectly reverses the tracker
     #       track_path!(x_root, lambda_root, -dx0, -dlambda0, utils, ws; stop_iters=steps_per_dir)
#
     #       println("\n")
        end
    end
end