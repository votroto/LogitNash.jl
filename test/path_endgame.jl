using LinearAlgebra
using PATHSolver
using ForwardDiff

# ---------- Game ----------
struct TensorGame{T<:Real,N}
    payoffs::Vector{Array{T,N}}   # length P, each size (m1,...,mP)
    m::Vector{Int}
end

function TensorGame(payoffs::Vector{A}) where {T<:Real,N,A<:AbstractArray{T,N}}
    P = length(payoffs)
    @assert P > 1
    dims = size(payoffs[1])
    @assert length(dims) == P
    for i in 2:P
        @assert size(payoffs[i]) == dims
    end
    TensorGame{T,N}(Array{T,N}[Array(payoffs[i]) for i in 1:P], collect(dims))
end

# ---------- Indexing ----------
struct NashIdx
    π_off::Vector{Int}   # starts for each player strategy block
    v_off::Vector{Int}   # indices of values
    Mπ::Int              # total number of strategy vars
    M::Int               # total vars = Mπ + P
end

function make_idx(m::Vector{Int})
    P = length(m)
    π_off = zeros(Int, P)
    k = 1
    for i in 1:P
        π_off[i] = k
        k += m[i]
    end
    Mπ = k - 1
    v_off = collect(Mπ+1:Mπ+P)
    NashIdx(π_off, v_off, Mπ, Mπ + P)
end

@inline π_block(i, m, idx::NashIdx) = idx.π_off[i]:(idx.π_off[i] + m[i] - 1)

function π_views(z::AbstractVector, m::Vector{Int}, idx::NashIdx)
    P = length(m)
    xs = Vector{SubArray{eltype(z),1,typeof(z),Tuple{UnitRange{Int}},true}}(undef, P)
    for i in 1:P
        xs[i] = @view z[π_block(i,m,idx)]
    end
    xs
end
# generic expected_action_values
function expected_action_values(game::TensorGame, i::Int, xs::Vector)
    P  = length(game.m)
    mi = game.m[i]
    Ui = game.payoffs[i]

    T = eltype(xs[1])                  # Float64 or Dual
    g = zeros(T, mi)

    for I in CartesianIndices(Ui)
        t = Tuple(I)
        ai = t[i]
        prob = one(T)
        @inbounds for j in 1:P
            if j != i
                prob *= xs[j][t[j]]
            end
        end
        g[ai] += T(Ui[I]) * prob
    end
    return g
end

# generic F_map!
function F_map!(F::AbstractVector{T}, z::AbstractVector{T}, game::TensorGame, idx::NashIdx) where {T}
    m = game.m
    P = length(m)
    xs = π_views(z, m, idx)
    v  = @view z[idx.v_off]
    fill!(F, zero(T))

    k = 1
    for i in 1:P
        g = expected_action_values(game, i, xs)
        for a in 1:m[i]
            F[k] = v[i] - g[a]
            k += 1
        end
    end
    for i in 1:P
        F[k] = sum(xs[i]) - one(T)
        k += 1
    end
    return F
end

# J via ForwardDiff now works
function J_dense!(J::Matrix{Float64}, z::Vector{Float64}, game::TensorGame, idx::NashIdx)
    f = x -> begin
        out = similar(x)               # keeps Dual type when needed
        F_map!(out, x, game, idx)
        out
    end
    J .= ForwardDiff.jacobian(f, z)
    return J
end
# PATH callbacks
function F_cb(n::Int32, x::Vector{Float64}, f::Vector{Float64}, game::TensorGame, idx::NashIdx)
    @assert Int(n) == length(x) == length(f)
    F_map!(f, x, game, idx)
    return Cint(0)
end

function J_cb(n::Int32, nnz::Int32, x::Vector{Float64},
              col::Vector{Int32}, len::Vector{Int32},
              row::Vector{Int32}, data::Vector{Float64},
              game::TensorGame, idx::NashIdx)
    N = Int(n)
    J = zeros(Float64, N, N)
    J_dense!(J, x, game, idx)

    # dense column-compressed structure
    k = 1
    for j in 1:N
        col[j] = Int32(k)
        len[j] = Int32(N)
        for i in 1:N
            row[k] = Int32(i)
            data[k] = J[i,j]
            k += 1
        end
    end
    @assert k - 1 == Int(nnz)
    return Cint(0)
end

function solve_nash_path(game::TensorGame; warmstart_π::Vector{Vector{Float64}})
    m = game.m
    P = length(m)
    idx = make_idx(m)
    M = idx.M

    @assert length(warmstart_π) == P
    z0 = zeros(Float64, M)

    # warm-start π
    for i in 1:P
        @assert length(warmstart_π[i]) == m[i]
        r = π_block(i, m, idx)
        xi = max.(warmstart_π[i], 0.0)
        s = sum(xi)
        z0[r] .= (s > 0 ? xi ./ s : fill(1.0/m[i], m[i]))
    end

    # warm-start v_i = max_a g_i[a] at warm π
    xs0 = π_views(z0, m, idx)
    for i in 1:P
        g = expected_action_values(game, i, xs0)
        z0[idx.v_off[i]] = maximum(g)
    end

    # bounds
    lb = fill(-Inf, M)
    ub = fill( Inf, M)

    # π >= 0
    for i in 1:P
        lb[π_block(i,m,idx)] .= 0.0
    end
    # v free (already -Inf..Inf)

    status, z, info = PATHSolver.solve_mcp(
        (n, x, f) -> F_cb(n, x, f, game, idx),
        (n, nnz, x, col, len, row, data) -> J_cb(n, nnz, x, col, len, row, data, game, idx),
        lb, ub, z0;
        nnz = M*M,
        silent = true,
        jacobian_structure_constant = true
    )

    πstar = [copy(z[π_block(i,m,idx)]) for i in 1:P]
    # tiny cleanup
    for i in 1:P
        πstar[i] .= max.(πstar[i], 0.0)
        s = sum(πstar[i]); if s > 0; πstar[i] ./= s; end
    end
    vstar = copy(z[idx.v_off])

    return status, πstar, vstar, z, info
end