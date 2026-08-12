using BenchmarkTools
using LinearAlgebra




function build_deriv_loops_old(dims, idx, prev_p, p, q, N)
    d = dims[idx]
    var_ad = Symbol("a", d)
    var_ap = Symbol("a", p)
    var_aq = Symbol("a", q)

    if idx == length(dims)
        # --- INNERMOST LOOP (always d = 1) ---
        u_args = [Symbol("a", k) for k in 1:N]
        pay_idx = Expr(:ref, :pay_p, u_args...)

        if d == p || d == q
            # Case A: Innermost loop varies an output index.
            # Safe to write directly to res_pq.
            body = if prev_p === nothing
                quote
                    @inbounds res_pq[$var_ap, $var_aq] += $pay_idx
                end
            else
                quote
                    @inbounds res_pq[$var_ap, $var_aq] += $pay_idx * $prev_p
                end
            end

            return quote
                @simd for $var_ad in 1:size(pay_p, $d)
                    $body
                end
            end
        else
            # Case B: Innermost loop is a marginalized dimension.
            # Output indices are constant. Use a scalar accumulator for SIMD!
            p_term = prev_p === nothing ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])

            return quote
                s = zero(eltype(res_pq))
                @simd for $var_ad in 1:size(pay_p, $d)
                    @inbounds s += $pay_idx * $p_term
                end
                @inbounds res_pq[$var_ap, $var_aq] += s
            end
        end
    else
        # --- OUTER LOOPS ---
        if d == p || d == q
            # Skip probability multiplication for output dimensions
            inner_loop = build_deriv_loops_old(dims, idx + 1, prev_p, p, q, N)

            return quote
                for $var_ad in 1:size(pay_p, $d)
                    $inner_loop
                end
            end
        else

            # Multiply probability for marginalized dimensions
            new_p = Symbol("p", d)
            p_expr = prev_p === nothing ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])
            inner_loop = build_deriv_loops_old(dims, idx + 1, new_p, p, q, N)

            return quote
                for $var_ad in 1:size(pay_p, $d)
                    $new_p = $p_expr
                    $inner_loop
                end
            end
        end
    end
end

@generated function unilateral_derivatives_old!(
    result::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    exprs = []

    for p in 1:N
        for q in 1:N
            p == q && continue
            dims = N:-1:1
            inner_loops_expr = build_deriv_loops_old(dims, 1, nothing, p, q, N)

            push!(exprs, quote
                res_pq = result[$p][$q]
                pay_p = payoffs[$p]
                fill!(res_pq, zero(eltype(res_pq)))
                $inner_loops_expr
            end)
        end
    end

    return Expr(:block, exprs...)
end

















function build_deriv_loops_simpler(d, last_w, p, q, N)
    var_ad = Symbol("a", d)
    var_ap = Symbol("a", p)
    var_aq = Symbol("a", q)

    if d == 1
        actions = [Symbol("a", k) for k in 1:N]
        pay_idx = Expr(:ref, :payoff, actions...)

        # Use a scalar accumulator for SIMD for marginalized dimension.
        if 1 == p || 1 == q
            return quote
                @simd for $var_ad in axes(payoff, $d)
                    result[$var_ap, $var_aq] += $pay_idx * $last_w
                end
            end
        else
            return quote
                s = zero(eltype(result))
                @simd for $var_ad in axes(payoff, $d)
                    s += $pay_idx * $last_w * pi[$d][$var_ad]
                end
                result[$var_ap, $var_aq] += s
            end
        end
    else
        new_w = Symbol("w", d)
        w_expr = (d == p || d == q) ? :($last_w) : :($last_w * pi[$d][$var_ad])
        inner_loop = build_deriv_loops_simpler(d - 1, new_w, p, q, N)

        return quote
            for $var_ad in axes(payoff, $d)
                $new_w = $w_expr
                $inner_loop
            end
        end
    end
end

@generated function unilateral_derivatives_simpler!(
    results::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    exprs = []

    for p in 1:N
        for q in 1:N
            p == q && continue
            dims = N:-1:1
            inner_loops_expr = build_deriv_loops_simpler(N, 1, p, q, N)

            push!(exprs, quote
                result = results[$p][$q]
                payoff = payoffs[$p]
                fill!(result, zero(eltype(result)))
                @inbounds $inner_loops_expr
            end)
        end
    end

    return Expr(:block, exprs...)
end#= ========================================================== =#












function build_outer_fused_loops(d, prev_exprs, p, N)
    var_ad = Symbol("a", d)

    if d == 1
        # --- INNERMOST LOOPS (Unfused) ---
        u_args = [Symbol("a", k) for k in 1:N]
        pay_idx = Expr(:ref, :pay_p, u_args...)
        inner_blocks = Expr[]

        for q in 1:N
            q == p && continue
            var_ap = Symbol("a", p)
            var_aq = Symbol("a", q)
            res_sym = Symbol("res_p", p, "_q", q)
            prev_q = prev_exprs[q]

            if 1 == p || 1 == q
                # Case A: Safe to write directly to res_pq
                push!(inner_blocks, quote
                    @simd for $var_ad in 1:size(pay_p, 1)
                        @inbounds $res_sym[$var_ap, $var_aq] += $pay_idx * $prev_q
                    end
                end)
            else
                # Case B: Scalar accumulator
                s_sym = Symbol("s_q", q)
                p_term = :($prev_q * pi[1][$var_ad])
                push!(inner_blocks, quote
                    $s_sym = zero(eltype($res_sym))
                    @simd for $var_ad in 1:size(pay_p, 1)
                        @inbounds $s_sym += $pay_idx * $p_term
                    end
                    @inbounds $res_sym[$var_ap, $var_aq] += $s_sym
                end)
            end
        end

        return Expr(:block, inner_blocks...)
    else
        # --- OUTER LOOPS (Fused) ---
        setup_exprs = Expr[]
        new_exprs = Dict{Int,Any}()

        for q in 1:N
            q == p && continue
            prev_q = prev_exprs[q]
            new_sym = Symbol("p_d", d, "_q", q)
            new_exprs[q] = new_sym

            # Reverted to your exact, clean logic. LLVM will optimize away the 1 * x
            p_expr = (d == p || d == q) ? :($prev_q) : :($prev_q * pi[$d][$var_ad])
            push!(setup_exprs, :($new_sym = $p_expr))
        end

        inner_loops = build_outer_fused_loops(d - 1, new_exprs, p, N)

        return quote
            for $var_ad in 1:size(pay_p, $d)
                $(setup_exprs...)
                $inner_loops
            end
        end
    end
end

@generated function unilateral_derivatives_optimal!(
    result::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    exprs = Expr[]

    for p in 1:N
        res_setup = Expr[]
        for q in 1:N
            q == p && continue
            res_sym = Symbol("res_p", p, "_q", q)
            push!(res_setup, quote
                $res_sym = result[$p][$q]
                fill!($res_sym, zero(eltype($res_sym)))
            end)
        end

        # Initial products are identically 1
        prev_exprs = Dict{Int,Any}(q => 1 for q in 1:N if q != p)
        outer_loops_expr = build_outer_fused_loops(N, prev_exprs, p, N)

        push!(exprs, quote
            $(res_setup...)
            pay_p = payoffs[$p]
            $outer_loops_expr
        end)
    end

    return Expr(:block, exprs...)
end#= ========================================================= =#








function build_fused_body(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        # Keep separate @simd loops per q so LLVM can vectorize contiguous 1D memory
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
        # Outer loops: only assign weights for dimensions that aren't p or q
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
                $(build_fused_body(d - 1, next_w, p, N))
            end
        end
    end
end

@generated function unilateral_derivatives_optimal_simplified!(
    results::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    p_blocks = map(1:N) do p
        init_w = Any[1 for _ in 1:N]
        body = build_fused_body(N, init_w, p, N)
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

function unilateral_derivatives_optimal_simplified_body!(
    results::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    p_blocks = map(1:N) do p
        init_w = Any[1 for _ in 1:N]
        body = build_fused_body(N, init_w, p, N)
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

#=
function build_deriv_loops(dims, idx, prev_p, p, q_target, players, N)
    d = dims[idx]
    var_ad = Symbol("a_", d)

    if idx == length(dims)
        # --- INNERMOST LOOP (always d=1, mapping to player p) ---
        # Perfect contiguous access: res_pq[a_1, ...] varying a_1
        u_args = [Symbol("a_", k) for k in 1:N]
        pay_idx = Expr(:ref, :pay_p, u_args...)

        # Find which dimension corresponds to the target q
        var_aq = Symbol("a_", findfirst(==(q_target), players))

        body = prev_p === nothing ?
            :(res_pq[a_1, $var_aq] += $pay_idx) :
            :(res_pq[a_1, $var_aq] += $pay_idx * $prev_p)

        return quote
            @simd for a_1 in axes(pay_p, 1)
                @inbounds $body
            end
        end
    else
        # --- OUTER LOOPS (d > 1) ---
        q = players[d]
        if q == q_target
            # This is the derivative target, do not multiply by pi
            inner_loop = build_deriv_loops(dims, idx + 1, prev_p, p, q_target, players, N)
            return quote
                for $var_ad in axes(pay_p, $d)
                    $inner_loop
                end
            end
        else
            # Marginalized dimension
            new_p = Symbol("p_", d)
            p_expr = prev_p === nothing ? :(pi[$q][$var_ad]) : :($prev_p * pi[$q][$var_ad])
            inner_loop = build_deriv_loops(dims, idx + 1, new_p, p, q_target, players, N)
            return quote
                for $var_ad in axes(pay_p, $d)
                    $new_p = $p_expr
                    $inner_loop
                end
            end
        end
    end
end

@generated function unilateral_derivatives!(
    result::NTuple{N,NTuple{N,Matrix}},
    U_perm::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where {N}
    exprs = []

    for p in 1:N
        players = [p; [k for k in 1:N if k != p]]
        for q in 1:N
            p == q && continue
            dims = N:-1:1
            inner_loops_expr = build_deriv_loops(dims, 1, nothing, p, q, players, N)

            push!(exprs, quote
                res_pq = result[$p][$q]
                pay_p = U_perm[$p]
                fill!(res_pq, zero(eltype(res_pq)))
                $inner_loops_expr
            end)
        end
    end

    return Expr(:block, exprs...)
end
=#


function unilateral_derivatives_3p!(
    result::NTuple{3,NTuple{3,Matrix{T}}},
    payoffs::NTuple{3,Array{T,3}},
    pi::NTuple{3,Vector{T}}
) where T
    U1, U2, U3 = payoffs
    pi1, pi2, pi3 = pi
    A1, A2, A3 = size(U1)

    # --- Player 1 ---
    res12 = result[1][2];
    fill!(res12, zero(T))
    res13 = result[1][3];
    fill!(res13, zero(T))

    @inbounds for a3 in 1:A3
        p3 = pi3[a3]
        for a2 in 1:A2
            p2 = pi2[a2]

            # Loop 1: Pulls U1 slice into L1 cache, writes to res12
            @simd ivdep for a1 in 1:A1
                res12[a1, a2] += U1[a1, a2, a3] * p3
            end

            # Loop 2: Reads U1 slice from L1 cache, writes to res13
            @simd ivdep for a1 in 1:A1
                res13[a1, a3] += U1[a1, a2, a3] * p2
            end
        end
    end

    # --- Player 2 ---
    res21 = result[2][1];
    fill!(res21, zero(T))
    res23 = result[2][3];
    fill!(res23, zero(T))

    @inbounds for a3 in 1:A3
        p3 = pi3[a3]
        for a2 in 1:A2

            @simd ivdep for a1 in 1:A1
                res21[a2, a1] += U2[a1, a2, a3] * p3
            end

            s23 = zero(T)
            @simd ivdep for a1 in 1:A1
                s23 += U2[a1, a2, a3] * pi1[a1]
            end
            res23[a2, a3] += s23
        end
    end

    # --- Player 3 ---
    res31 = result[3][1];
    fill!(res31, zero(T))
    res32 = result[3][2];
    fill!(res32, zero(T))

    @inbounds for a3 in 1:A3
        for a2 in 1:A2
            p2 = pi2[a2]

            @simd ivdep for a1 in 1:A1
                res31[a3, a1] += U3[a1, a2, a3] * p2
            end

            s32 = zero(T)
            @simd ivdep for a1 in 1:A1
                s32 += U3[a1, a2, a3] * pi1[a1]
            end
            res32[a3, a2] += s32
        end
    end

    return nothing
end











using LoopVectorization

function unilateral_derivatives_3p_turbo!(
    result::NTuple{3,NTuple{3,Matrix{T}}},
    payoffs::NTuple{3,Array{T,3}},
    pi::NTuple{3,Vector{T}}
) where {T}
    U1, U2, U3 = payoffs
    pi1, pi2, pi3 = pi
    A1, A2, A3 = size(U1)

    # Player 1
    fill!(result[1][2], zero(T))
    fill!(result[1][3], zero(T))
    fill!(result[2][1], zero(T))
    fill!(result[2][3], zero(T))
    fill!(result[3][1], zero(T))
    fill!(result[3][2], zero(T))

    @turbo for a_3 in 1:A3, a_2 in 1:A2, a_1 in 1:A1
        result[1][2][a_1, a_2] += U1[a_1, a_2, a_3] * pi3[a_3]
        result[1][3][a_1, a_3] += U1[a_1, a_2, a_3] * pi2[a_2]
    end
    @turbo for a_3 in 1:A3, a_2 in 1:A2, a_1 in 1:A1

        result[2][1][a_2, a_1] += U2[a_1, a_2, a_3] * pi3[a_3]
        result[2][3][a_2, a_3] += U2[a_1, a_2, a_3] * pi1[a_1]
    end
    @turbo for a_3 in 1:A3, a_2 in 1:A2, a_1 in 1:A1

        result[3][1][a_3, a_1] += U3[a_1, a_2, a_3] * pi2[a_2]
        result[3][2][a_3, a_2] += U3[a_1, a_2, a_3] * pi1[a_1]
    end

    return nothing
end






A = 150
D = 3
dims = ntuple(_ -> A, D)
Us = ntuple(_ -> randn(dims...), D);

pi = ntuple(_ -> normalize(rand(A), 1), D);

dudpi1 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D)
dudpi2 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D)

unilateral_derivatives_3p!(dudpi1, Us, pi)
unilateral_derivatives_optimal_simplified!(dudpi2, Us, pi)

@show norm.(dudpi1[i] .- dudpi2[i] for i in eachindex(dudpi1))

nothing

Us = ntuple(_ -> randn(dims...), D);

a = @benchmark unilateral_derivatives_3p!($dudpi2, $Us, data) setup=(data=ntuple(_ -> normalize(rand(A), 1), D))
b = @benchmark unilateral_derivatives_optimal_simplified!($dudpi1, $Us, data) setup=(data=ntuple(_ -> normalize(rand(A), 1), D))


#=


A = 5
D = 6
dims = ntuple(_ -> A, D);
Us = ntuple(_ -> randn(dims...), D);
U_perm = ntuple(D) do p
    # Move player p to dim 1, keeping remaining dimensions sequential
    permutedims(Us[p], [p; [q for q in 1:D if q != p]])
end;
pi = ntuple(_ -> normalize(rand(A),1), D);

dudpi1 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D);
dudpi2 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D);

unilateral_derivatives_old!(dudpi1, Us, pi);
unilateral_derivatives!(dudpi2, U_perm, pi);

@show norm.(dudpi1[i] .- dudpi2[i] for i in eachindex(dudpi1))

nothing

=#

#=


@benchmark unilateral_derivatives_old!($dudpi1, $Us, $pi)
@benchmark unilateral_derivatives!($dudpi2, $U_perm, $pi)


=#
