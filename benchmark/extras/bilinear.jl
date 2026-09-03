using LinearAlgebra
using Gurobi
using JuMP

function fischer_gupte(us::NTuple{2,Matrix}; optimizer=Gurobi.Optimizer)
    m = Model(optimizer)

    nx, ny = size(us[1])
    @variable(m, xs[1:nx], lower_bound=0, upper_bound=1)
    @variable(m, ys[1:ny], lower_bound=0, upper_bound=1)
    @variable(m, w[i=1:2], lower_bound=minimum(us[i]), upper_bound=maximum(us[i]))

    @constraint(m, sum(xs) == 1)
    @constraint(m, sum(ys) == 1)

    @constraint(m, dot(xs, us[1], ys) + dot(xs, us[2], ys) >= sum(w))
    @constraint(m, (us[1] * ys) .<= w[1])
    @constraint(m, (xs' * us[2]) .<= w[2])

    optimize!(m)

    if termination_status(m) == JuMP.OPTIMAL
        (value.(xs), value.(ys)), solve_time(m)
    else
        nothing, NaN
    end
end

function cerny_gupta_kroer_no_warm(us::NTuple{2,Matrix{Float64}}; optimizer=Gurobi.Optimizer)
    m = Model(optimizer)

    n1, n2 = size(us[1])

    @variable(m, varpi)
    @variable(m, minimum(us[1]) <= v1 <= maximum(us[1]))
    @variable(m, minimum(us[2]) <= v2 <= maximum(us[2]))
    @variable(m, 0 <= d1[1:n1] <= 1)
    @variable(m, 0 <= d2[1:n2] <= 1)
    @constraint(m, sum(d1) == 1)
    @constraint(m, sum(d2) == 1)

    @constraint(m, v1 .>= us[1] * d2)
    @constraint(m, v2 .>= us[2]' * d1)

    @constraint(m, varpi .>= d1 .* (v1 .- us[1] * d2))
    @constraint(m, varpi .>= d2 .* (v2 .- us[2]' * d1))

    @constraint(m, varpi .>= -d1 .* (v1 .- us[1] * d2))
    @constraint(m, varpi .>= -d2 .* (v2 .- us[2]' * d1))

    @objective(m, Min, varpi)

    optimize!(m)

    if termination_status(m) == JuMP.OPTIMAL
        (value.(d1), value.(d2)), solve_time(m)
    else
        nothing, NaN
    end
end