module GameTheoryExt

using LogitNash
using GameTheory: AbstractNormalFormGame, NormalFormGame, nums_actions

function _extract_tensors(game::NormalFormGame{N}) where N
    invperm(p::Int) = ntuple(i -> i + ifelse(i < p, N-p+1, 1-p), Val(N))
    ntuple(p -> permutedims(game.players[p].payoff_array, invperm(p)), Val(N))
end

function _extract_tensors(game::AbstractNormalFormGame{N,R}) where {N,R}
    dims = nums_actions(game)
    Us = ntuple(_ -> Array{R,N}(undef, dims), Val(N))
    @inbounds for I in CartesianIndices(dims)
        u = game[I]
        for p in eachindex(Us)
            Us[p][I] = u[p]
        end
    end
    return Us
end

"""
    solve(game::AbstractNormalFormGame{N,R}; kwargs...) where {N,R<:Real}

Extension for `GameTheory.AbstractNormalFormGame` types.
"""
function LogitNash.solve(
    game::AbstractNormalFormGame{N,R};
    kwargs...
) where {N,R<:Real}
    solve(_extract_tensors(game); kwargs...)
end

end