include("utils.jl")

using Revise
using LogitNash
using Random

A = 5
D = 6
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

_pi, _status = solve(Us; stop_lambda=10.0, stop_eps=1e-2)

Uss = [ntuple(_ -> randn(ntuple(_ -> A, D)...), D) for _ in 1:20]
pis = nothing
@time for i in 1:20
    global pis
    pz, status = solve(Uss[i]; stop_lambda=Inf, stop_eps=1e-6)
    pis = pz
end

Random.seed!(3462345634)

A = 5
D = 5

Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D)

ne, st = solve(Us)

@show st
show_profile(ne)