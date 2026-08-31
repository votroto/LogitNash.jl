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

function print_strats(ss)
    for p in eachindex(ss)
        println(round.(ss[p]; digits=5))
    end
end


A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pi,status = solve(Us)


Random.seed!(3462345634)


A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

@time pi,status = solve(Us)

print_strats(pi)
@show status
