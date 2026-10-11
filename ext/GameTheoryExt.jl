module GameTheoryExt

using LogitNash
using GameTheory: NormalFormGame

function _extract_tensors(game::NormalFormGame{N}) where N
    invperm(p::Int) = ntuple(i -> i + ifelse(i < p, N-p+1, 1-p), N)
    ntuple(p -> permutedims(game.players[p].payoff_array, invperm(p)), N)
end

"""
    solve(game::NormalFormGame{N,R}; kwargs...) where {N,R<:Real}

Extension for `GameTheory.NormalFormGame`.
"""
function LogitNash.solve(game::NormalFormGame{N,R}; kwargs...) where {N,R<:Real}
    solve(_extract_tensors(game); kwargs...)
end

end