include("../test/utils.jl")
using LogitNash
using Printf
using LinearAlgebra

function timed_solve(data; stop_lambda=Inf, stop_eps=1e-8)
    t_parse = @timed tensors = LogitNash.parse_nfg(data)
    t_solve = @timed ne, status = solve(tensors; stop_lambda, stop_eps)
    gap = equilibrium_gap_precise(tensors, ne)

    err = status.lambda < stop_lambda && status.regret > stop_eps || status.stall
    return t_parse, t_solve, err, gap
end

function main(cmd, num_samples)
    data = read(cmd)
    timed_solve(data)

    for _ in 1:num_samples
        try
            data = read(cmd)
            GC.gc(false)
            tp, ts, err, gap  = timed_solve(data)
            @printf "%.4e %d %.4e\n" (tp.time + ts.time) err gap
        catch e
            @error e
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    num_samples = parse(Int, ARGS[1])
    cmd = `sh -c $(ARGS[2])`

    main(cmd, num_samples)
end
