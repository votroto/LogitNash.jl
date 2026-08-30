using LogitNash
using Printf

# some of the games are quite big, allow enough iterations
stop_iters=10000
stop_lambda=Inf
stop_eps=1e-6

function randn_game(num_players, num_actions)
    dims = ntuple(_ -> num_actions, num_players)
    ntuple(_ -> randn(dims...), num_players)
end

min_actions = 2
max_actions = 10
increment_actions = 1
min_players = 2
max_players = 9
increment_players = 1
num_samples = 5
# So that they fit into L3
max_game_size = 8*10^6

print("|       | ")
for num_actions in min_actions:increment_actions:max_actions
    @printf "%3d act. | " num_actions
end
println()
print("|------:|")
for num_actions in min_actions:increment_actions:max_actions
    print("---------:|")
end
println()

for num_players in min_players:increment_players:max_players
    print("| **$num_players** | ")
    for num_actions in min_actions:increment_actions:max_actions
        if 8 * num_players * Float64(num_actions)^num_players >= max_game_size
            print("         | ")
            continue
        end

        game = randn_game(num_players, num_actions)
        _ne, _status = LogitNash.solve(game; stop_lambda, stop_eps)

        total = 0.0
        sampl = 0
        for _ in 1:num_samples
            game = randn_game(num_players, num_actions)
            t_solve = @timed ne, status = LogitNash.solve(game; stop_lambda, stop_eps, stop_iters)
            if status.regret <= stop_eps
                total += t_solve.time
                sampl += 1
            else
                @error status
            end
        end

        @printf "%8.4f | " total / num_samples
    end
    println()
end
println()