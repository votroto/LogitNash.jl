using Revise

using LogitNash
using Printf

const LAMBDA::Float64 = 1e6
const REGRET::Float64 = 1e-6
const ITERATIONS::Int = 1000

function timed_solve(data; stop_lambda, stop_eps, stop_iters)
    t_parse = @timed tensors = LogitNash.parse_nfg(data)
    t_solve = @timed ne, status = solve(tensors; stop_lambda, stop_eps, stop_iters)

    return t_parse, t_solve, status
end

function interpret_status(status)
    if status.stall
        'S', false
    elseif status.lambda < 0.0
        'N', false
    elseif status.iteration >= ITERATIONS
        'I', false
    elseif status.lambda < LAMBDA && status.regret > REGRET
        '?', false
    else
        '.', true
    end
end

function timed_solve_file(gamefile)
    try
        data = read(gamefile)
        tp, ts, status = timed_solve(data; stop_lambda=LAMBDA, stop_eps=REGRET, stop_iters=ITERATIONS)

        sym, ok = interpret_status(status)

        ts.time - ts.compile_time + tp.time - tp.compile_time, sym, ok
    catch e
        @error "Failed to solve gamefile" e
        NaN, '?', false
    end
end

function solve_datasets(root::String)
    items = readdir(root, join=true)
    subdirs = filter(isdir, items)
    gamefiles = filter(isfile, items)

    for subdir in subdirs
        solve_datasets(subdir)
    end

    if isempty(gamefiles)
        return
    end

    total_time = 0.0
    solved_games = 0
    dataset_fails = String[]

    GC.gc(false)

    for gamefile in gamefiles
        tim, sym, ok = timed_solve_file(gamefile)

        print(sym)
        if ok
            total_time += tim
            solved_games += 1
        else
            push!(dataset_fails, gamefile)
        end
    end

    @printf "\nAvg solvetime for dataset '%s' (%d): %.1f ms\n" root solved_games (total_time * 1000 / solved_games)

    if !isempty(dataset_fails)
        @warn "Failed games in dataset '$root'" dataset_fails
    end
    println()
end

# solve_datasets("nfgs")