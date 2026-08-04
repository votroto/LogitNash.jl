include("../src/spaces.jl")
include("../src/turocy.jl")
include("../src/path.jl")

using Random
using LinearAlgebra

function export_gambit_format_buf(payoffs)
    players = eachindex(payoffs)

    player_string = join(["\"$i\"" for i in players], " ")
    dim_string = join(size(first(payoffs)), " ")

    ioin = IOBuffer()

    println(ioin, "NFG 1 R \"Exported Game\"")
    println(ioin, "{ $player_string } { $dim_string }")

    for i in eachindex(payoffs[1])
        for p in eachindex(payoffs)
            print(ioin, payoffs[p][i])
            print(ioin, " ")
        end
    end

    seekstart(ioin)
    ioin
end

function parse_gambit_output(line, payoffs::NTuple{N}) where N
    players = eachindex(payoffs)
    dims = size(first(payoffs))

    ne = parse.(Float64, split(strip(line), ",")[2:end])
    nes = Vector{Vector{Float64}}(undef, length(players))
    offset = 1
    for (i, p) in enumerate(players)
        nes[i] = @view ne[offset:(offset+dims[p]-1)]
        offset += dims[p]
    end

    return ntuple(i -> nes[i], N)
end

function gambit_equilibrium(
    in::IOBuffer
)
    open(pipeline(`gambit-logit -q -e -l1e6`; stdin=in), "r", stdout) do ioout
        zz = read(ioout, String)
        return zz
    end
end

function covariant_game(actions::NTuple{N,Int}, r) where {N}
    @assert -1/(N-1) <= r <= 1

    Σ = Matrix{Float64}(I, N, N)
    for i in 1:N, j in 1:N
        if i != j
            Σ[i, j] = r
        end
    end

    L = cholesky(Symmetric(Σ)).L

    G = Array{Float64}(undef, actions..., N)

    for I in CartesianIndices(actions)
        @views G[Tuple(I)..., :] .= L * randn(N)
    end

    slices = eachslice(G, dims=N)
    ntuple(i -> Array(slices[i]), N)
end

using Random
Random.get_tls_seed()

#Random.seed!(3462315634)
#=
Us = (
    randn(2, 2, 2, 2, 2),
    randn(2, 2, 2, 2, 2),
    randn(2, 2, 2, 2, 2),
    randn(2, 2, 2, 2, 2),
    randn(2, 2, 2, 2, 2)
)

nash(Us; stop_eps=0.0)
io = export_gambit_format_buf(Us)
gambit_equilibrium(io)
=#

function runtest(n, seed)
    Random.seed!(seed)
    Us = (
        randn(n, n, n, n, n),
        randn(n, n, n, n, n),
        randn(n, n, n, n, n),
        randn(n, n, n, n, n),
        randn(n, n, n, n, n)
    )

    tj = @timed pi, status = nash(Us; stop_eps=0.0)

        if status.regret >= 1e-4 || status.t <= 100 || status.stall ==true # !all(norm.(pi .- pig) .<= 1e-5)
            @show status
            println(seed)
        end
    #
    return true
    io = export_gambit_format_buf(Us)
    tc = @timed pigstr = gambit_equilibrium(io)
    pig = parse_gambit_output(pigstr, Us)

    agree = all(norm.(pi .- pig) .<= 1e-5)
    eqg = check_equilibrium(Us, pig)
    eqj = check_equilibrium(Us, pi)

    println("$(tj.time) $(tc.time) $agree $eqg $eqj")
    pi, pig
end

function print_strats(ss)
    for p in eachindex(ss)
        println(round.(ss[p]; digits=5))
    end
end

function unilateral_deviations_simple(
    payoffs::NTuple{N,<:AbstractArray{P,N}},
    x::NTuple{N,<:AbstractVector{X}}
) where {N,P,X}
    T = promote_type(P, X)

    result = ntuple(i -> zeros(T, size(payoffs[i], i)), N)
    for i in CartesianIndices(first(payoffs))
        @simd for p in 1:N
            @inbounds w = prod(x[z][i[z]] for z in 1:N if z != p)
            @inbounds result[p][i[p]] += w * payoffs[p][i]
        end
    end
    result
end

function check_equilibrium(
    payoffs::NTuple{N,<:AbstractArray{P,N}},
    pi::NTuple{N,<:AbstractVector{X}}
) where {N,P,X}
    deviations = unilateral_deviations_simple(payoffs, pi)
    max_deviation_incentive(deviations, pi)
end

using Random
Random.seed!(34873478)

A::Matrix{Float64} = zeros(2,2) #*floatmin()
B::Matrix{Float64} = zeros(2,2) #*floatmin()

payoffs = (A, B)

@show pi, status = nash(payoffs; stop_eps=-1.0, stop_iters=10000, stop_t=Inf)


print_strats(pi)
#=
io = export_gambit_format_buf(payoffs)
tc = @timed pigstr = gambit_equilibrium(io)
pig = parse_gambit_output(pigstr, payoffs)

println()
print_strats(pig)



=#