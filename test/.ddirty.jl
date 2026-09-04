
using Revise
using LogitNash
using Random
using LinearAlgebra

function scramble_game(game::NTuple{N}, player_perm, action_perms) where N
    return ntuple(N) do p
        orig_player = player_perm[p]
        permutedims(game[orig_player][action_perms...], player_perm)
    end
end

function scramble_eq(eq::NTuple{N}, player_perm, action_perms) where N
    return ntuple(N) do p
        orig_player = player_perm[p]
        eq[orig_player][action_perms[orig_player]]
    end
end

function solve_gl(tensors)
    iobuf = IOBuffer()
    LogitNash.dump_nfg(iobuf, tensors)
    seekstart(iobuf)

    cmd = `gambit-logit -qe -l1000000`
    output = read(pipeline(cmd, stdin=iobuf), String)
    nes = parse.(Float64, split(strip(output), ',')[2:end])
    ne = collect.(LogitNash.splitviews(nes, size(first(tensors))))

    ne
end

function solve_ln(tensors; stop_lambda=1e6, stop_eps=NaN)
    ne, stat = solve(tensors; stop_lambda, stop_eps)
    ne
end


println()

for (path, dirs, files) in walkdir("nfgs")
    println("# $path")

    for f in files

        data = read(joinpath(path, f))
        tensors = LogitNash.parse_nfg(data)

        pi_orig = solve_gl(tensors)

        for perm in 1:5
            player_perm = shuffle(eachindex(tensors))
            action_perm = ntuple(k -> shuffle(axes(tensors[k], k)), length(tensors))

            payoffs_perm = scramble_game(tensors, player_perm, action_perm)
            expected = scramble_eq(pi_orig, player_perm, action_perm)

            pi_perm = solve_gl(payoffs_perm)

            if all((<=)(1e-6), norm.(expected .- pi_perm))
                print("_")
            else
                print("\e[31m", '#', "\e[0m")
            end
        end
        print(" ")
    end
    println()
end
