# Starts from random points to try and walk all the zero-residual paths present
# in a game. The output is tailored for Gnuplot. The path output is implemented
# by parsing the debug log.

using Revise
using LogitNash
using LinearAlgebra
using Printf
using Logging

"""Used for dumping the path debug information to a gnuplot readable format"""
struct PathLogger <: AbstractLogger
    stream::IO
end

function Logging.handle_message(logger::PathLogger, _lvl, _msg, _mod, _grp, id, _file, _ln; pi, lambda)
    if id == :end
        println(logger.stream, "\n")
    elseif id == :step
        dump_profile(logger.stream, pi)
        println(logger.stream, " ", lambda)
    end
end

Logging.shouldlog(::PathLogger, _lvl, _mod, group, id) = id in (:step, :end) && group == :tracker
Logging.min_enabled_level(::PathLogger) = Logging.Debug

function dump_profile(io, profile)
    format_strat(strat) = join((@sprintf "%.4e" a for a in strat), " ")
    print(io, join((format_strat(s) for s in profile), " "))
end

dump_profile(profile) = dump_profile(stdout, profile)


function rand_range(t_min, t_max)
    rand() * (t_max - t_min) + t_min
end

function prob_to_redlograt(y::AbstractVector)
    x = similar(y, length(y)-1)
    for i in eachindex(x)
        x[i] = log(y[i] / y[end])
    end
    x
end

function guess_x(dims)
    vcat(map(d -> prob_to_redlograt(normalize(rand(d), 1)), dims)...)
end

function reset_refs(ws, dims)
    @. ws.refs = dims
end

function explore_solutions(
    utils::NTuple{N};
    num_starts::Int=200,
    stop_iters::Int=200,
) where {N}
    dims = size(first(utils))

    _x = LogitNash.uniform_xprofile(dims)

    _dx = zero(_x)
    dt = 1.0
    ds = 0.01

    _ws = LogitNash.make_hc_workspace(_x, dims)
    ws = LogitNash.make_hc_workspace(_x, dims)
    dx = zero(_x)

    for iter in 0:num_starts
        x_guess = iter == 0 ? _x : guess_x(dims)
        t_guess = iter == 0 ? 0.0 : rand_range(1.0, 8.0)

        reset_refs(_ws, dims)

        st_corr, x, t = LogitNash.correct!(_ws.x_nxt, x_guess, t_guess, _dx, dt, utils, _ws; abs_tol=1e-9, max_iters=200)

        if st_corr != LogitNash.STATUS_SUCCESS
            continue
        end

        LogitNash.update_predictor_jacobian!(x, t, _dx, dt, utils, _ws)
        dx, dt = LogitNash.predict_direction!(dx, _ws)

        reset_refs(ws, dims)
        LogitNash._solve!(utils, 1*x, t, +1*dx, +1*dt, ds, ws; stop_iters)

        reset_refs(ws, dims)
        LogitNash._solve!(utils, 1*x, t, -1*dx, -1*dt, ds, ws; stop_iters)
    end
end

# Example 3p-2a game with a tricky hairpin-shaped path.

hairpin222 = (
    [0.3 0.89; 0.6 0.7;;; 0.78 0.79; 0.78 0.5],
    [0.22 0.68; 0.4 0.81;;; 0.91 0.85; 0.72 0.12],
    [0.77 0.01; 0.15 0.2;;; 0.79 0.02; 0.46 0.22]
)

A = 2
D = 3

Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D)

open("/tmp/path.dat", "w") do io
    logger = PathLogger(io)

    with_logger(logger) do
        @time explore_solutions(hairpin222)
    end

end
