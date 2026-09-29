include("cerny_impl.jl")
include("../../../test/utils.jl")

using Printf
using Revise
using LogitNash
using Random
using LinearAlgebra

function timed_solve_cg(data)
    tensors = LogitNash.parse_nfg(data)
    cgne, cg_time = cerny(tensors)
    cg_gap = equilibrium_gap_precise(tensors, cgne)

    cg_time, cg_gap
end

function all_finite(dat)
    tensors = LogitNash.parse_nfg(dat)

    for U in tensors
        for u in U
            if !isfinite(u)
                return false
            end
        end
    end
    return true
end

function solve_cmd(cmd, num_samples)

    cg_ts, cg_gs = fill(NaN, num_samples), fill(NaN, num_samples)

    for i in 1:num_samples
        data = nothing

        while isnothing(data)
            try
                redirect_stderr(devnull) do
                    data = read(cmd)
                end
                if !all_finite(data)
                    @warn "broken game"
                    data = nothing
                end
            catch
            end
        end

        try
            GC.gc(false)
            redirect_stdout(devnull) do

                cg_time, cg_gap = timed_solve_cg(data)

                cg_ts[i] = cg_time
                cg_gs[i] = cg_gap
            end
        catch e
            @error e
        end
    end

    println("# cg (times then gaps)")
    for i in 1:num_samples
        @printf "%.4e %.4e\n" cg_ts[i] cg_gs[i]
    end

end

for game_size in 31:40
    cmd = `java -jar $(ENV["HOME"])/opt/gamut.jar -random_params -players 2 -actions $game_size -normalize -min_payoff -1 -max_payoff 1 -output GambitOutput -f /dev/stdout -g RandomGame`

    println("# RandomGame size 2 x $game_size")
    solve_cmd(cmd, 50)

    println()
    println()
end