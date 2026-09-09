using Printf

function show_profile(io, profile)
    format_strat(strat) = join((@sprintf "%6.4f" a for a in strat), ", ")
    println(io, join(("[$(format_strat(s))]" for s in profile), "\n"))
end

show_profile(profile) = show_profile(stdout, profile)

function unilateral_derivatives_simple(
    payoffs::NTuple{N,Array{R,N}},
    pi::NTuple{N,Vector{Float64}}
) where {N,R}
    dims = ntuple(i -> size(payoffs, i), Val(N))
    results = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), Val(N)), Val(N))
    for i in CartesianIndices(first(payoffs))
        for p in 1:N, q in 1:N
            p==q && continue
            w = prod(pi[b][i[b]] for b in 1:N if b != p && b != q)
            results[p][q][i[p], i[q]] += w * payoffs[p][i]
        end
    end
    results
end

function unilateral_deviations_simple(
    payoffs::NTuple{N,Array},
    xs::NTuple{N,Vector}
) where N
    dims = ntuple(i -> size(payoffs, i), Val(N))
    result = ntuple(i -> zeros(dims[i]), Val(N))
    for i in CartesianIndices(first(payoffs))
        for p in 1:N
            w = prod(xs[q][i[q]] for q in 1:N if q != p)
            result[p][i[p]] += w * payoffs[p][i]
        end
    end
    result
end

function _max_deviation_incentive(
    deviations::NTuple{N,Vector},
    xs::NTuple{N,Vector}
) where N
    actuals = dot.(deviations, xs)
    bests = maximum.(deviations)

    maximum(bests[p] - actuals[p] for p in 1:N)
end

function equilibrium_gap_precise(
    payoffs::NTuple{N,Array},
    _xs::NTuple{N,Vector}
) where N
    xs = ntuple(i -> normalize(BigFloat.(_xs[i]), 1), N)

    deviations = _unilateral_deviations_simple(payoffs, xs)
    _max_deviation_incentive(deviations, xs)
end