using Revise
using LogitNash
include("../benchmark/parse_nfg.jl")
using Printf

function timed_solve(data; stop_lambda=1e6, stop_eps=0.0,stop_iters=1000)
    t_parse = @timed tensors = parse_nfg(data)
    t_solve = @timed ne, status = LogitNash.solve(tensors; stop_lambda, stop_eps, stop_iters)

    err = status.lambda < stop_lambda && status.regret > stop_eps || status.stall
   # if err
    ##    @show status
  #  end
    return t_parse, t_solve, status
end

fails = String[]
for dataset in readdir("nfgs", join=true)
    global fails
    data = read(first(readdir(dataset, join=true)))
    tp, ts, err = timed_solve(data)
    tottim = 0.0
    totga = 0
    for gamefile in readdir(dataset, join=true)
        try
            data = read(gamefile)
            GC.gc()
            tp, ts, status = timed_solve(data)
            tottim += ts.time
            totga += 1
            fff = true
            if status.stall
                print('S')
            elseif status.lambda < 0.0
                print('N')
            elseif status.iteration >= 1000
                print('I')
            elseif status.lambda < 1e6 && status.regret > 0.0
                print('?')
            else
                fff=false
                print('.')
            end


            if fff
                push!(fails, gamefile)
            end
        catch e
            @error e
        end
    end
    println()
    @printf "Avg solvetime for dataset '%s' (%d): %.1f ms\n" dataset totga (tottim*1000/totga)
    println()
end

println("failed games:")
println.(fails)
nothing