

function build_deriv_loops(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        a_all = [Symbol("a", k) for k in 1:N]
        ap = Symbol("a", p)

        q_loops = map(1:N) do q
            q == p && return :()
            aq = Symbol("a", q)
            res = :(results[$p][$q])
            wq = w[q]

            if p == 1 || q == 1
                quote
                    @simd ivdep for a1 in 1:size(pay_p, 1)
                        $res[$ap, $aq] += pay_p[$(a_all...)] * $wq
                    end
                end
            else
                quote
                    s = zero(eltype($res))
                    @simd ivdep for a1 in 1:size(pay_p, 1)
                        s += pay_p[$(a_all...)] * $wq * pi[1][a1]
                    end
                    $res[$ap, $aq] += s
                end
            end
        end
        return Expr(:block, q_loops...)
    else
        assignments = Expr[]
        next_w = copy(w)

        for q in 1:N
            if q != p && d != p && d != q
                w_sym = Symbol("w_d", d, "_q", q)
                push!(assignments, :($w_sym = $(w[q]) * pi[$d][$ad]))
                next_w[q] = w_sym
            end
        end

        return quote
            for $ad in 1:size(pay_p, $d)
                $(assignments...)
                $(build_deriv_loops(d - 1, next_w, p, N))
            end
        end
    end
end

@generated function unilateral_derivatives!(
    results::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    p_blocks = map(1:N) do p
        init_w = Any[1 for _ in 1:N]
        body = build_deriv_loops(N, init_w, p, N)
        quote
            pay_p = payoffs[$p]
            $body
        end
    end

    return quote
        for p in 1:N, q in 1:N
            p != q && fill!(results[p][q], zero(eltype(results[p][q])))
        end
        @inbounds begin
            $(p_blocks...)
        end
    end
end






function build_deriv_loops_fast(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        a_all = [Symbol("a", k) for k in 1:N]
        ap = Symbol("a", p)

        if p == 1
            # Contiguous writes for q != 1
            writes = [:( (results[1][$q])[a1, $(Symbol("a", q))] += val * $(w[q]) ) for q in 2:N]

            return quote
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    $(writes...)
                end
            end
        else
            # Shared scalar reductions for all q > 1
            accumulators = [:( (results[$p][$q])[$ap, $(Symbol("a", q))] += s_shared * $(w[q]) ) for q in 2:N if q != p]

            return quote
                s_shared = zero(T)
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    (results[$p][1])[a1, $ap] += val * $(w[1])
                    s_shared += val * pi[1][a1]
                end
                $(accumulators...)
            end
        end
    end

    # Helper to clearly define which dimensions get hoisted constants
    is_active(q) = (q != p && d != p && d != q)
    next_w = Any[is_active(q) ? Symbol("w_d", d, "_q", q) : w[q] for q in 1:N]
    assignments = [:( $(next_w[q]) = $(w[q]) * pi[$d][$ad] ) for q in 1:N if is_active(q)]

    return quote
        for $ad in axes(pay_p, $d)
            $(assignments...)
            $(build_deriv_loops_fast(d - 1, next_w, p, N))
        end
    end
end

@generated function unilateral_derivatives_fast!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    p_blocks = map(1:N) do p
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops_fast(N, init_w, p, N)
        quote
            pay_p = payoffs[$p]
            $body
        end
    end

    return quote
        for p in 1:N, q in 1:N
            p != q && fill!(results[p][q], zero(T))
        end
        @inbounds begin
            $(p_blocks...)
        end
    end
end




function jacobian_x!(J, pi, lam, dudpi, u::NTuple{N}) where {N}
    eq_i = 1

    @inbounds for eq_p in eachindex(u)
        for eq_a in 1:(size(u[eq_p], eq_p)-1)
            pd_i = 1

            for pd_p in eachindex(u)
                if pd_p == eq_p
                    # Own-player identity block
                    for pd_a in 1:(size(u[eq_p], eq_p)-1)
                        J[eq_i, pd_i] = (eq_a == pd_a)

                        pd_i += 1
                    end
                else
                    d = dudpi[eq_p][pd_p]
                    transposed = eq_p != 1 && pd_p == 1

                    c = 0.0
                    for pd_a in eachindex(pi[pd_p])
                        gm = transposed ?
                            (d[pd_a, eq_a] - d[pd_a, end]) :
                            (d[eq_a, pd_a] - d[end, pd_a])

                        c += gm * pi[pd_p][pd_a]
                    end

                    for pd_a in 1:(size(u[pd_p], pd_p)-1)
                        gm = transposed ?
                            (d[pd_a, eq_a] - d[pd_a, end]) :
                            (d[eq_a, pd_a] - d[end, pd_a])

                        J[eq_i, pd_i] =
                            -lam * pi[pd_p][pd_a] * (gm - c)

                        pd_i += 1
                    end
                end
            end

            eq_i += 1
        end
    end

    return J
end



function unilateral_deviations_from_derivatives2!(
    out::NTuple{N,Vector},
    dudpi::NTuple{N,NTuple{N,Matrix}},
    pi::NTuple{N,Vector},
) where N
    for p in 1:N
        # We can just pick the first available opponent index to contract out
        q = p == 1 ? 2 : 1

        # dudpi[p][q] * pi[q] directly yields the length-actions_p vector
        mul!(out[p], dudpi[p][q], pi[q])
    end

    return out
end

function jacobian_x2!(J, pi, lam, dudpi, u::NTuple{N}) where {N}
    eq_i = 1

    @inbounds for eq_p in eachindex(u)
        for eq_a in 1:(size(u[eq_p], eq_p)-1)
            pd_i = 1

            for pd_p in eachindex(u)
                if pd_p == eq_p
                    for pd_a in 1:(size(u[eq_p], eq_p)-1)
                        J[eq_i, pd_i] = (eq_a == pd_a)
                        pd_i += 1
                    end
                else
                    d = dudpi[eq_p][pd_p]
                    c = 0.0
                    for pd_a in eachindex(pi[pd_p])
                        gm = d[eq_a, pd_a] - d[end, pd_a]
                        c += gm * pi[pd_p][pd_a]
                    end

                    for pd_a in 1:(size(u[pd_p], pd_p)-1)
                        gm = d[eq_a, pd_a] - d[end, pd_a]
                        J[eq_i, pd_i] = -lam * pi[pd_p][pd_a] * (gm - c)
                        pd_i += 1
                    end
                end
            end

            eq_i += 1
        end
    end

    return J
end




















function build_deriv_loops_fast2(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        a_all = [Symbol("a", k) for k in 1:N]
        ap = Symbol("a", p)

        if p == 1
            # Contiguous writes for q != 1
            writes = [:( (results[1][$q])[a1, $(Symbol("a", q))] += val * $(w[q]) ) for q in 2:N]

            return quote
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    $(writes...)
                end
            end
        else
            # Shared scalar reductions for all q > 1
            accumulators = [:( (results[$p][$q])[$ap, $(Symbol("a", q))] += s_shared * $(w[q]) ) for q in 2:N if q != p]

            return quote
                s_shared = zero(T)
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]

                    # ↓ CHANGED: Accumulate contiguously into results[1][p] buffer instead of results[p][1]
                    (results[1][$p])[a1, $ap] += val * $(w[1])

                    s_shared += val * pi[1][a1]
                end
                $(accumulators...)
            end
        end
    end

    # Helper to clearly define which dimensions get hoisted constants
    is_active(q) = (q != p && d != p && d != q)
    next_w = Any[is_active(q) ? Symbol("w_d", d, "_q", q) : w[q] for q in 1:N]
    assignments = [:( $(next_w[q]) = $(w[q]) * pi[$d][$ad] ) for q in 1:N if is_active(q)]

    return quote
        for $ad in axes(pay_p, $d)
            $(assignments...)
            $(build_deriv_loops_fast2(d - 1, next_w, p, N))
        end
    end
end

using LinearAlgebra

@generated function unilateral_derivatives_fast2!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    p_blocks = map(1:N) do p
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops_fast2(N, init_w, p, N)

        if p == 1
            quote
                pay_p = payoffs[$p]
                $body
            end
        else
            quote
                pay_p = payoffs[$p]
                $body

                # Move the contiguous workspace data into the correctly-oriented matrix
                LinearAlgebra.transpose!(results[$p][1], results[1][$p])
                # Clean up the buffer so it's safely zeroed for when the p=1 block runs
                fill!(results[1][$p], zero(T))
            end
        end
    end

    return quote
        for p in 1:N, q in 1:N
            p != q && fill!(results[p][q], zero(T))
        end

        @inbounds begin
            # 1. Run all p > 1 blocks first. They will temporarily hijack results[1][p]
            #    to achieve lightning-fast contiguous memory access.
            $([p_blocks[p] for p in 2:N]...)

            # 2. Run the p = 1 block last. Because we just re-zeroed results[1][p],
            #    it writes the actual derivative data safely into clean arrays.
            $(p_blocks[1])
        end
    end
end





function unilateral_derivatives_fast2_body!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    p_blocks = map(1:N) do p
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops_fast2(N, init_w, p, N)

        if p == 1
            quote
                pay_p = payoffs[$p]
                $body
            end
        else
            quote
                pay_p = payoffs[$p]
                $body

                # Move the contiguous workspace data into the correctly-oriented matrix
                LinearAlgebra.transpose!(results[$p][1], results[1][$p])
                # Clean up the buffer so it's safely zeroed for when the p=1 block runs
                fill!(results[1][$p], zero(T))
            end
        end
    end

    return quote
        for p in 1:N, q in 2:N
            p != q && fill!(results[p][q], zero(T))
        end

        @inbounds begin
            # 1. Run all p > 1 blocks first. They will temporarily hijack results[1][p]
            #    to achieve lightning-fast contiguous memory access.
            $([p_blocks[p] for p in 2:N]...)

            # 2. Run the p = 1 block last. Because we just re-zeroed results[1][p],
            #    it writes the actual derivative data safely into clean arrays.
            $(p_blocks[1])
        end
    end
end






function jacobian_x3!(J, pi, lam, dudpi, u::NTuple{N}) where {N}
    eq_start = 1

    @inbounds for eq_p in 1:N
        num_actions_eq = size(u[eq_p], eq_p)
        A_eq = num_actions_eq - 1

        A_eq == 0 && continue

        pd_start = 1
        for pd_p in 1:N
            num_actions_pd = size(u[pd_p], pd_p)
            A_pd = num_actions_pd - 1

            A_pd == 0 && continue

            if pd_p == eq_p
                # 1. Own-player identity block (Perfectly Column-Major)
                for pd_a in 1:A_pd
                    J_col = pd_start + pd_a - 1
                    @simd ivdep for eq_a in 1:A_eq
                        J_row = eq_start + eq_a - 1
                        J[J_row, J_col] = ifelse(eq_a == pd_a, one(eltype(J)), zero(eltype(J)))
                    end
                end
            else
                d = dudpi[eq_p][pd_p]
                pi_pd = pi[pd_p]

                # --- ZERO ALLOCATION TRICK ---
                # We temporarily hijack the first column of the Jacobian block
                # we are currently building to store the expected payoffs `c`.
                J_col_buf = pd_start

                for eq_a in 1:A_eq
                    J[eq_start + eq_a - 1, J_col_buf] = zero(eltype(J))
                end

                # Accumulate `c` across all opponent actions into our hijacked column
                for pd_a in 1:num_actions_pd
                    p_val = pi_pd[pd_a]
                    d_end = d[num_actions_eq, pd_a]

                    @simd ivdep for eq_a in 1:A_eq
                        J_row = eq_start + eq_a - 1
                        J[J_row, J_col_buf] += (d[eq_a, pd_a] - d_end) * p_val
                    end
                end

                # --- BUILD JACOBIAN (Perfectly Column-Major) ---
                # Populate columns 2 through A_pd, reading `c` from our buffer column
                for pd_a in 2:A_pd
                    J_col = pd_start + pd_a - 1
                    p_val_lam = -lam * pi_pd[pd_a]
                    d_end = d[num_actions_eq, pd_a]

                    @simd ivdep for eq_a in 1:A_eq
                        J_row = eq_start + eq_a - 1
                        c = J[J_row, J_col_buf]
                        gm = d[eq_a, pd_a] - d_end

                        J[J_row, J_col] = p_val_lam * (gm - c)
                    end
                end

                # Finally, compute column 1, safely overwriting our buffer with the real answer!
                p_val_lam_1 = -lam * pi_pd[1]
                d_end_1 = d[num_actions_eq, 1]

                @simd ivdep for eq_a in 1:A_eq
                    J_row = eq_start + eq_a - 1
                    c = J[J_row, J_col_buf]
                    gm = d[eq_a, 1] - d_end_1

                    J[J_row, J_col_buf] = p_val_lam_1 * (gm - c)
                end
            end

            pd_start += A_pd
        end

        eq_start += A_eq
    end

    return J
end




















@inline function _jac_block_4!(
    J, d, π, λ,
    r0, c0, np, Dp, Dq,
)
    i = 1

    # Four-row tiles.
    while i + 3 <= np
        c1 = zero(eltype(J))
        c2 = zero(eltype(J))
        c3 = zero(eltype(J))
        c4 = zero(eltype(J))

        # -------------------------------------------------------------
        # c[r] = Σ_k (d[r,k] - d[Dp,k]) * π[k]
        #
        # k is outer, rows are contiguous in d.
        # -------------------------------------------------------------
        for k in 1:Dq
            πk = π[k]
            dD = d[Dp, k]

            c1 += (d[i,     k] - dD) * πk
            c2 += (d[i + 1, k] - dD) * πk
            c3 += (d[i + 2, k] - dD) * πk
            c4 += (d[i + 3, k] - dD) * πk
        end

        # -------------------------------------------------------------
        # Form the four rows of the block.
        #
        # j outer, i inner => contiguous J stores and d loads.
        # -------------------------------------------------------------
        for j in 1:(Dq - 1)
            πj = π[j]
            dD = d[Dp, j]

            J[r0 + i - 1, c0 + j - 1] =
                -λ * πj * (d[i,     j] - dD - c1)

            J[r0 + i,     c0 + j - 1] =
                -λ * πj * (d[i + 1, j] - dD - c2)

            J[r0 + i + 1, c0 + j - 1] =
                -λ * πj * (d[i + 2, j] - dD - c3)

            J[r0 + i + 2, c0 + j - 1] =
                -λ * πj * (d[i + 3, j] - dD - c4)
        end

        i += 4
    end

    # Cleanup: at most three rows.
    while i <= np
        c = zero(eltype(J))

        for k in 1:Dq
            c += (d[i, k] - d[Dp, k]) * π[k]
        end

        for j in 1:(Dq - 1)
            J[r0 + i - 1, c0 + j - 1] =
                -λ * π[j] *
                (d[i, j] - d[Dp, j] - c)
        end

        i += 1
    end

    return nothing
end

function jacobian_x_chatgpt!(
    J,
    pi,
    lam,
    dudpi,
    u::NTuple{N},
) where {N}

    row0 = 1
    col0 = 1

    @inbounds for p in 1:N
        Dp = size(u[p], p)
        np = Dp - 1

        col0 = 1

        for q in 1:N
            Dq = size(u[q], q)
            nq = Dq - 1

            if p == q
                # Identity block.
                for j in 1:np
                    for i in 1:np
                        J[row0 + i - 1, col0 + j - 1] = (i == j)
                    end
                end
            else
                _jac_block_4!(
                    J,
                    dudpi[p][q],
                    pi[q],
                    lam,
                    row0,
                    col0,
                    np,
                    Dp,
                    Dq,
                )
            end

            col0 += nq
        end

        row0 += np
    end

    return J
end

using LinearAlgebra















# Handles the inner loop when p == 1
@inline function compute_inner_p1!(results, pay_p::Array{T,N}, w::NTuple{N,T}, a_tail::CartesianIndex) where {N,T}
    @simd ivdep for a1 in axes(pay_p, 1)
        @inbounds val = pay_p[a1, a_tail]

        # Because N is known at compile time via the NTuple signature,
        # LLVM will fully unroll this loop just like your macro did.
        for q in 2:N
            @inbounds results[1][q][a1, a_tail[q-1]] += val * w[q]
        end
    end
end

# Handles the inner loop when p > 1
@inline function compute_inner_p_gt_1!(results, pay_p::Array{T,N}, pi_1::Vector{T}, p::Int, w::NTuple{N,T}, a_tail::CartesianIndex) where {N,T}
    s_shared = zero(T)
    ap = a_tail[p-1]

    @simd ivdep for a1 in axes(pay_p, 1)
        @inbounds val = pay_p[a1, a_tail]
        @inbounds results[1][p][a1, ap] += val * w[1]
        @inbounds s_shared += val * pi_1[a1]
    end

    for q in 2:N
        if q != p
            @inbounds results[p][q][ap, a_tail[q-1]] += s_shared * w[q]
        end
    end
end

function build_deriv_loops_refactored(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        # Pack the runtime variables into structures the functions can take
        w_expr = :( tuple($(w...)) )
        a_tail_expr = :( CartesianIndex($( (Symbol("a", k) for k in 2:N)... )) )

        if p == 1
            return :( compute_inner_p1!(results, pay_p, $w_expr, $a_tail_expr) )
        else
            return :( compute_inner_p_gt_1!(results, pay_p, pi[1], $p, $w_expr, $a_tail_expr) )
        end
    end

    # Helper to clearly define which dimensions get hoisted constants
    is_active(q) = (q != p && d != p && d != q)
    next_w = Any[is_active(q) ? Symbol("w_d", d, "_q", q) : w[q] for q in 1:N]
    assignments = [:( $(next_w[q]) = $(w[q]) * pi[$d][$ad] ) for q in 1:N if is_active(q)]

    return quote
        for $ad in axes(pay_p, $d)
            $(assignments...)
            $(build_deriv_loops_refactored(d - 1, next_w, p, N))
        end
    end
end




@generated function unilateral_derivatives_refactored!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    p_blocks = map(1:N) do p
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops_refactored(N, init_w, p, N)

        if p == 1
            quote
                pay_p = payoffs[$p]
                $body
            end
        else
            quote
                pay_p = payoffs[$p]
                $body

                # Move the contiguous workspace data into the correctly-oriented matrix
                LinearAlgebra.transpose!(results[$p][1], results[1][$p])
                # Clean up the buffer so it's safely zeroed for when the p=1 block runs
                fill!(results[1][$p], zero(T))
            end
        end
    end

    return quote
        for p in 1:N, q in 1:N
            p != q && fill!(results[p][q], zero(T))
        end

        @inbounds begin
            # 1. Run all p > 1 blocks first. They will temporarily hijack results[1][p]
            #    to achieve lightning-fast contiguous memory access.
            $([p_blocks[p] for p in 2:N]...)

            # 2. Run the p = 1 block last. Because we just re-zeroed results[1][p],
            #    it writes the actual derivative data safely into clean arrays.
            $(p_blocks[1])
        end
    end
end






using BenchmarkTools

N_dim = 10
S_size = 4
T = Float64

# Dummy Data
results1 = ntuple(p -> ntuple(q -> zeros(T, S_size, S_size), N_dim), N_dim)
results2 = ntuple(p -> ntuple(q -> zeros(T, S_size, S_size), N_dim), N_dim)

payoffs = ntuple(p -> rand(T, ntuple(_ -> S_size, N_dim)...), N_dim)
pi_vec = ntuple(p -> rand(T, S_size), N_dim)

# === 1. Test Compile Times ===

println("Compiling Refactored...")
@time unilateral_derivatives_refactored!(results2, payoffs, pi_vec)
println("Compiling Original...")
@time unilateral_derivatives_fast2!(results1, payoffs, pi_vec)

# Assert correctness
#@assert all(results1 .≈ results2) "Math output diverges!"
@show ([ norm.(results1[i] .- results2[i]) for i in eachindex(results1)])

# === 2. Test Runtime Performance ===
println("\nBenchmarking Original...")
@btime unilateral_derivatives_fast2!($results1, $payoffs, $pi_vec)

println("\nBenchmarking Refactored...")
@btime unilateral_derivatives_refactored!($results2, $payoffs, $pi_vec)

















#=
using BenchmarkTools



A = 5
D = 5
dims = ntuple(_ -> A, D);
Us = ntuple(_ -> randn(dims...), D);

pi = ntuple(_ -> normalize(rand(A),1), D);

dudpi1 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D);
dudpi2 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D);


D = length(dims) # Or N
dudpi1_transposed = ntuple(p -> ntuple(q -> (q == 1) ? zeros(dims[q], dims[p]) : zeros(dims[p], dims[q]), D), D)


unilateral_derivatives_fast2!(dudpi1, Us, pi);
unilateral_derivatives_fast!(dudpi1_transposed, Us, pi);

@time dudpi2 = ntuple(p -> ntuple(q ->
    (q == 1) ? collect(transpose(dudpi1_transposed[p][q])) : dudpi1_transposed[p][q],
D), D)

@show ([ norm.(dudpi1[i] .- dudpi2[i]) for i in eachindex(dudpi1)])


=#
#b = @benchmark unilateral_derivatives_fast!($dudpi1_transposed, $Us, $pi)
#a = @benchmark unilateral_derivatives_fast2!($dudpi1, $Us, $pi)

#n = sum(d-1 for d in dims)
#Fx = zeros(n, n)
#Fx2 = zeros(n, n)
#@benchmark jacobian_x2!($Fx2, $pi, 0.3, $dudpi1, Us)
#@benchmark jacobian_x_chatgpt!($Fx, $pi, 0.3, $dudpi1, Us)


#d = @benchmark jacobian_x_chatgpt!($Fx2, $pi, $(0.5), $dudpi1, $Us)
#c = @benchmark jacobian_x3!($Fx, $pi, $(0.5), $dudpi1, $Us)

#unilateral_derivatives_fast2_body!(dudpi1, Us, pi);