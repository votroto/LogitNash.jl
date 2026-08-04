include("../src/LogitNash.jl")
include("parse_nfg.jl")
using Printf

function timed_solve(data)
    time_parse = @timed tensors = parse_nfg(data)
    time_solve = @timed ne, status = LogitNash.nash(tensors; stop_t=1e6, stop_eps=1e-6)

    if status.t <= 100 && status.regret >= 1e-4 && status.stall == true
        @error "Possible non-convergence!\n$status"
    end
    return time_parse, time_solve, status
end

if abspath(PROGRAM_FILE) == @__FILE__
    num_samples = isempty(ARGS) ? 1 : parse(Int, ARGS[1])

    for cmdstr in eachline(stdin)
        @info cmdstr
        cmd = `sh -c $cmdstr`
        println("# $cmdstr")
        data = read(cmd)
        timed_solve(data)

        for _ in 1:num_samples
            try
                data = read(cmd)
                GC.gc()
                tp, ts, status = timed_solve(data)
                @printf "%.4e %.4e\n" tp.time ts.time
            catch e
                @error e
            end
        end
        print("\n\n")
    end
end
