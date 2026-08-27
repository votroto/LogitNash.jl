using Revise
using LogitNash
using Random
using LinearAlgebra


function print_strats(ss)
    for p in eachindex(ss)
        println(round.(ss[p]; digits=5))
    end
end

function unilateral_deviations_simple(
    payoffs::NTuple{N,Array{Float64,N}},
    xs::NTuple{N,Vector{Float64}}
) where N
    result = ntuple(i -> zeros(size(payoffs[i], i)), N)
    for i in CartesianIndices(first(payoffs))
        for p in 1:N
            w = prod(xs[q][i[q]] for q in 1:N if q != p; init=1.0)
            result[p][i[p]] += w * payoffs[p][i]
        end
    end
    result
end

function max_deviation_incentive(
    deviations::NTuple{N,Vector{Float64}},
    xs::NTuple{N,Vector{Float64}}
) where N
    actuals = dot.(deviations, xs)
    bests = maximum.(deviations)

    maximum(bests[p] - actuals[p] for p in 1:N)
end

function equilibrium_gap(
    payoffs::NTuple{N,Array{Float64,N}},
    xs::NTuple{N,Vector{Float64}}
) where N
    deviations = unilateral_deviations_simple(payoffs, xs)
    max_deviation_incentive(deviations, xs)
end


A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pi,status = nash(Us)


Random.seed!(3462345634)


A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

@time pi,status = nash(Us)

print_strats(pi)
@show status
