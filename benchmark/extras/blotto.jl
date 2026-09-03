using LogitNash
using Random
using LinearAlgebra

#=
// An implementation of the Blotto: https://en.wikipedia.org/wiki/Blotto_game
// This version supports n >= 2 players. Each player distributes M coins on N
// fields. Each field is won by at most one player: the one with the most
// coins on the specific field; if there is a draw, the field is considered
// drawn (not won by any player), and hence ignored in the scoring. The winner
// is the player with the most won fields: all player have won the same number
// of fields, they each receive 0. Otherwise, the winners share 1 / (number of
// winners) and losers share -1 / (number of losers), reducing to {-1,0,1} in
// the 2-player case.
//
// Parameters:
//   "coins"      int    number of coins each player starts with (default: 10)
//   "fields"     int    number of fields (default: 3)
//   "players"    int    number of players (default: 2)
=#

function generate_blotto_actions(coins::Int, ::Val{K}) where K
    actions = NTuple{K,Int}[]
    function generate(fields_left, coins_left, current)
        if fields_left == 1
            push!(actions, (current..., coins_left))
            return
        end
        for c in 0:coins_left
            generate(fields_left - 1, coins_left - c, (current..., c))
        end
    end
    generate(K, coins, ())
    return actions
end

@inline function get_points(allocations::NTuple{P,NTuple{K,Int}}) where {P,K}
    ntuple(Val(P)) do p
        pts = 0
        @inbounds for f in 1:K
            c = allocations[p][f]
            is_winner = true
            for other_p in 1:P
                other_p == p && continue
                if allocations[other_p][f] >= c
                    is_winner = false
                    break
                end
            end
            pts += is_winner
        end
        pts
    end
end

@inline function evaluate_blotto_game(allocations::NTuple{P,NTuple{K,Int}}) where {P,K}
    points = get_points(allocations)

    max_pts = maximum(points)
    num_winners = count(==(max_pts), points)

    if num_winners == P
        return ntuple(_ -> 0.0, Val(P))
    end

    win_util = 1.0 / num_winners
    lose_util = -1.0 / (P - num_winners)

    return map(pt -> pt == max_pts ? win_util : lose_util, points)
end

function generate_blotto_tensors(players::Int, coins::Int, fields::Int)
    return _generate_blotto_tensors(Val(players), coins, Val(fields))
end

function _generate_blotto_tensors(::Val{P}, coins::Int, ::Val{K}) where {P,K}
    actions = generate_blotto_actions(coins, Val(K))
    num_actions = length(actions)

    dims = ntuple(_ -> num_actions, Val(P))
    tensors = ntuple(_ -> zeros(Float32, dims), Val(P))

    @inbounds for idx in CartesianIndices(dims)
        allocs = ntuple(p -> actions[idx.I[p]], Val(P))
        utils = evaluate_blotto_game(allocs)

        for p in 1:P
            tensors[p][idx] = utils[p]
        end
    end

    return tensors
end

warm = ntuple(_ -> rand(Float32, 2, 2, 2, 2), 4)
ne, st = solve(warm; stop_lambda=Inf, stop_eps=1e-2)

blotto = generate_blotto_tensors(4, 10, 3)

rtol = 1e-6 * (maximum(first(blotto)) - minimum(first(blotto)))
@time ne, st = solve(blotto; stop_lambda=Inf, stop_eps=rtol)