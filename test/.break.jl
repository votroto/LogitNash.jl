include("utils.jl")

# Script for fuzzing the solver.

using Revise
using LogitNash
using Random

A = 2
D = 3
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D)

function round!(A; kwargs...)
    for i in eachindex(A)
        A[i] = round(A[i]; kwargs...)
    end
end

@time for i in 1:100000
    if i % 1000 == 0
        print(i, " ")
    end

    for p in 1:D
        rand!(Us[p])
        round!(Us[p]; digits=2)
    end

    ne, st = solve(Us; stop_lambda=Inf, stop_eps=1e-6)

    if st.regret > 1e-6 || st.lambda < 0 || st.stall
        println()
        println(Us)

    end
end
