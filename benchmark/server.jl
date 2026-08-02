include("../src/LogitNash.jl")
include("parse_nfg.jl")

using Printf
using Sockets

function serve_solver(addr, port; kwargs...)
    server = listen(addr, port)

    while true
        conn = accept(server)
        @async try
            data = read(conn)
            time_parse = @timed tensors = parse_nfg(data)
            time_solve = @timed ne, status = LogitNash.nash(tensors; kwargs...)

            for strat in ne
                for i in 1:(length(strat)-1)
                    p = @sprintf "%.5f" strat[i]
                    write(conn, p)
                    write(conn, " ")
                end
                p = @sprintf "%.5f" strat[end]
                write(conn, p)
                write(conn, "\n")
            end
            write(conn, "\n")
            write(conn, "# t = ", @sprintf("%.2e", status.t), "\n")
            write(conn, "# iteration = $(status.iteration)\n")
            write(conn, "# stall = $(status.stall)\n")
            write(conn, "# regret = ", @sprintf("%.2e", status.regret), "\n")
            write(conn, "# parse time = ", @sprintf("%.3f", time_parse.time), "\n")
            write(conn, "# solve time = ", @sprintf("%.3f", time_solve.time), "\n")
            write(conn, "\n")
        catch e
            @error e
        finally
            isopen(conn) && close(conn)
        end
    end
end