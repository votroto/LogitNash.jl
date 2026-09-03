include("bilinear.jl")

using Revise
using LogitNash
using Random
using LinearAlgebra

function export_gambit_format_buf(payoffs)
    players = eachindex(payoffs)

    player_string = join(["\"p$i\"" for i in players], " ")
    dim_string = join(size(first(payoffs)), " ")

    ioin = IOBuffer()

    println(ioin, "NFG 1 R \"Exported Game\"")
    println(ioin, "{ $player_string } { $dim_string }")

    for i in eachindex(payoffs[1])
        for p in eachindex(payoffs)
            print(ioin, payoffs[p][i])
            print(ioin, " ")
        end
    end

    seekstart(ioin)
    ioin
end


using LogitNash

function timed_solve_fg(data)
    tensors = LogitNash.parse_nfg(data)
    fgne, fg_time = fischer_gupte(tensors)
    fg_gap = equilibrium_gap_precise(tensors, fgne)

    fg_time, fg_gap
end

function timed_solve_gl(data;)
    tensors = LogitNash.parse_nfg(data)

    cmd = `gambit-logit -qe`
    time = @timed output = read(pipeline(cmd, stdin=IOBuffer(data)), String)
    nes = parse.(Float64, split(strip(output), ',')[2:end])
    ne = collect.(LogitNash.splitviews(nes, size(first(tensors))))
    gap = equilibrium_gap_precise(tensors, ne)

    time.time, gap
end

function timed_solve_ln(data; stop_lambda=Inf, stop_eps=1e-12)
    tensors = LogitNash.parse_nfg(data)

    ln_time = @timed lnne, lnstatus = solve(tensors; stop_lambda, stop_eps)

    ln_gap = equilibrium_gap_precise(tensors, lnne)

    return ln_time.time, ln_gap
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

    ln_ts, ln_gs = fill(NaN, num_samples), fill(NaN, num_samples)
    gl_ts, gl_gs = fill(NaN, num_samples), fill(NaN, num_samples)
    fg_ts, fg_gs = fill(NaN, num_samples), fill(NaN, num_samples)

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
                ln_time, ln_gap = timed_solve_ln(data)
                gl_time, gl_gap = timed_solve_gl(data)
                fg_time, fg_gap = timed_solve_fg(data)


                ln_ts[i] = ln_time
                gl_ts[i] = gl_time
                fg_ts[i] = fg_time
                ln_gs[i] = ln_gap
                gl_gs[i] = gl_gap
                fg_gs[i] = fg_gap
            end
        catch e
            @error e
        end
    end

    println("# ln gl fg (times then gaps)")
    for i in 1:num_samples
        @printf "%.4e %.4e %.4e %.4e %.4e %.4e\n" ln_ts[i] gl_ts[i] fg_ts[i] ln_gs[i] gl_gs[i] fg_gs[i]
    end

end

A = 5
D = 2
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pi,status = solve(Us;stop_lambda=Inf, stop_eps=1e-6);


for game_size in 2:30
    cmd = `java -jar $(ENV["HOME"])/opt/gamut.jar -random_params -players 2 -actions $game_size -normalize -min_payoff -1 -max_payoff 1 -output GambitOutput -f /dev/stdout -g RandomGame`

    println("# RandomGame size 2 x $game_size")
    solve_cmd(cmd, 50)

    println()
    println()
end