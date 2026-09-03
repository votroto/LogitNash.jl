using Printf

function show_profile(profile;)
    format_strat(strat) = join((@sprintf "%6.4f" a for a in strat), ", ")
    println(join(("[$(format_strat(s))]" for s in profile), "\n"))
end

function dump_profile(profile)
    format_strat(strat) = join((@sprintf "%.4e" a for a in strat), " ")
    println(join((format_strat(s) for s in profile), " "))
end

function _unilateral_deviations_simple(
    payoffs::NTuple{N,Array},
    xs::NTuple{N,Vector}
) where N
    result = ntuple(i -> zeros(size(payoffs[i], i)), N)
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