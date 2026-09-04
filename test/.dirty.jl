include("utils.jl")

using Revise
using LogitNash
using Random

A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pi, status = solve(Us; stop_lambda=1.0, stop_eps=1.0)


Random.seed!(3462345634)

A = 10
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pis = nothing
@time for i in 1:20
    global pis
    pz, status = solve(Us; stop_lambda=Inf, stop_eps=1e-6)
    pis = pz
end
show_profile(pis)
@show status
