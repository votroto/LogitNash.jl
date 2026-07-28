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

#=

ra = rand(Int64, 1000)
for i in 1:1000
    if i%10==0
        println(i)
    end
    runtest(5, ra[i])
end


 A::Matrix{Float64} = [3 0; 0 2]
    B::Matrix{Float64} = [2 0; 0 3]
    Us = (A, B)
    pi, status = nash(Us)


    io = export_gambit_format_buf(Us)
    tc = @timed pigstr = gambit_equilibrium(io)
    pig = parse_gambit_output(pigstr, Us)

for p in eachindex(pi)
    println("pi_$p = ", round.(pi[p]; digits=5))
end
=#
    #=


p1 = Float64[3 1; 5 2;;; 1 0; 2 1]
p2 = Float64[3 5; 1 2;;; 1 2; 0 1]
p3 = Float64[3 1; 1 0;;; 5 2; 2 1]
three_prisoners = (p1, p2, p3)

pi, status = nash(three_prisoners; stop_iters=1000, stop_t=1e6, stop_eps=1e-6)
=#
#ipi, ipig = runtest(5, seed_invert)

#    bpi, bpig = runtest(5, seed_bigbad)



#=
seed_disagree =  4219951271420067762

seed_invert = -1266070885943056040
seed_bigbad = 5605519460181864334
seed_slow = -8273187459698682571

Random.seed!(seed_invert)
n = 5
    Us = (
        randn(n,n,n,n,n),
        randn(n,n,n,n,n),
        randn(n,n,n,n,n),
        randn(n,n,n,n,n),
        randn(n,n,n,n,n)
    )

    tj = @timed pi, status = nash(Us; stop_eps=0.0, stop_iters=10000)


    io = export_gambit_format_buf(Us)
    tc = @timed pigstr = gambit_equilibrium(io)
    pig = parse_gambit_output(pigstr, Us)

    #ipi, ipig = runtest(5, seed_invert)

    #    bpi, bpig = runtest(5, seed_bigbad)
=#
#inversion

#=
[0.21543, 0.28827, 0.26213, 0.1211, 0.11307]
[0.0, 0.00699, 0.05883, 0.31778, 0.6164]
[0.0, 0.0, 0.0, 0.73337, 0.26663]
[0.32376, 0.0, 0.0, 0.34957, 0.32667]
[0.105, 0.45382, 0.0, 0.0, 0.44118]


=#


Us = (
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5)
)

pi, status = nash(Us; stop_iters=1000, stop_t=1e6, stop_eps=1e-6)

using Random
Random.seed!(3462345634)

Us = (
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5),
    randn(5, 5, 5, 5, 5)
)

@time pi, status = nash(Us; stop_iters=1000, stop_t=1e6, stop_eps=1e-6)

for p in eachindex(pi)
    println(round.(pi[p]; digits=5))
end