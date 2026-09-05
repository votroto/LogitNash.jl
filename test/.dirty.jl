include("utils.jl")

using Revise
using LogitNash
using Random

A = 5
D = 6
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pi, status = solve(Us; stop_lambda=10.0, stop_eps=1e-2)



Random.seed!(3462345634)

A = 5
D = 6

Uss = [ntuple(_ -> randn(ntuple(_ -> A, D)...), D) for _ in 1:20]
pis = nothing
@time for i in 1:20
    global pis
    pz, status = solve(Uss[i]; stop_lambda=Inf, stop_eps=1e-6)
    pis = pz
end
show_profile(pis)




Random.seed!(3462345634)

A = 10
D = 5

Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D)

@time ne, st = solve(Us; stop_lambda=Inf, stop_eps=1e-6)

@show st
show_profile(ne)
