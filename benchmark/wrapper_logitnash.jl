include("parse_nfg.jl")

using LogitNash
using Printf

function timed_solve(data; stop_lambda=1e6, stop_eps=NaN)
    t_parse = @timed tensors = parse_nfg(data)
    t_solve = @timed ne, status = LogitNash.solve(tensors; stop_lambda, stop_eps)

    err = status.lambda < stop_lambda && status.regret > stop_eps || status.stall
    return t_parse, t_solve, err
end

function main(cmd, num_samples)
    data = read(cmd)
    timed_solve(data)

    for _ in 1:num_samples
        try
            data = read(cmd)
            GC.gc(false)
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
