include("parse_nfg.jl")

using LogitNash
using Printf

function timed_solve(data; stop_t=1e6, stop_eps=1e-6)
    t_parse = @timed tensors = parse_nfg(data)
    t_solve = @timed ne, status = LogitNash.nash(tensors; stop_t, stop_eps)

    err = status.t < stop_t || status.regret > stop_eps || status.stall
    return t_parse, t_solve, err
end

function main(cmd, num_samples)
    data = read(cmd)
    timed_solve(data)

    for _ in 1:num_samples
        try
            data = read(cmd)
            GC.gc()
            tp, ts, err = timed_solve(data)
            @printf "%.6f %d\n" (tp.time + ts.time) err
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
