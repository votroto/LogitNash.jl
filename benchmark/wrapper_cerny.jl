include("extras/cerny/cerny_impl.jl")
include("../test/utils.jl")

using Printf
using Revise
using LogitNash
using Random
using LinearAlgebra



function timed_solve_cg(data)
    tensors = LogitNash.parse_nfg(data)
    cgne, cg_time = cerny(tensors)
    cg_gap = equilibrium_gap_precise(tensors, cgne)

    cg_time, 0, cg_gap
end

function main(cmd, num_samples)
    for _ in 1:num_samples
        try
            data = read(cmd)
            GC.gc(false)
            tim, err, gap = NaN, 1, NaN
            redirect_stdout(devnull) do
                tim, err, gap = timed_solve_cg(data)
            end
            @printf "%.4e %d %.4e\n" tim err gap
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
