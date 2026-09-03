include("utils.jl")

using Revise
using LogitNash
using Random

A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pi, status = solve(Us; stop_lambda=Inf, stop_eps=1e-6)


Random.seed!(3462345634)

A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

@time pi, status = solve(Us; stop_lambda=Inf, stop_eps=1e-6)

show_profile(pi)
@show status