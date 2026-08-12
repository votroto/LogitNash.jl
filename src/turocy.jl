function build_deriv_loops(dims, idx, prev_p, p, q, N)
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
            inner_loop = build_deriv_loops(dims, idx + 1, prev_p, p, q, N)

            return quote
                for $var_ad in 1:size(pay_p, $d)
                    $inner_loop
                end
            end
        else
            # Multiply probability for marginalized dimensions
            new_p = Symbol("p", d)
            p_expr = prev_p === nothing ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])
            inner_loop = build_deriv_loops(dims, idx + 1, new_p, p, q, N)

            return quote
                for $var_ad in 1:size(pay_p, $d)
                    $new_p = $p_expr
                    $inner_loop
                end
            end
        end
    end
end

@generated function unilateral_derivatives!(
    result::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    exprs = []

    for p in 1:N
        for q in 1:N
            p == q && continue
            dims = N:-1:1
            inner_loops_expr = build_deriv_loops(dims, 1, nothing, p, q, N)

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
end

function unilateral_deviations_from_derivatives!(
    out::NTuple{N,Vector},
    dudpi::NTuple{N,NTuple{N,Matrix}},
    pi::NTuple{N,Vector},
) where N
    for p in 1:N
        q = p == 1 ? 2 : 1
        mul!(out[p], dudpi[p][q], pi[q])
    end

    return out
end

@generated function unilateral_deviations!(
    out::NTuple{N,Vector},
    U::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where {N}
    exprs = []

    for i in 1:N
        # Symbols for indices: a_1, a_2, ..., a_N
        indices = [Symbol("a_", k) for k in 1:N]

        # --- 1. INNERMOST LOOP (d = 1) ---
        if i == 1
            # Player 1: a_1 is the target dimension, accumulate directly into out_1
            loop_expr = quote
                @simd for a_1 in axes(U_i, 1)
                    out_i[a_1] += U_i[$(indices...)] * p_2
                end
            end
        else
            # Players i > 1: a_1 is NOT the target. Accumulate into a scalar register first!
            loop_expr = quote
                val_1 = zero(eltype(out_i))
                @simd for a_1 in axes(U_i, 1)
                    val_1 += U_i[$(indices...)] * pi[1][a_1]
                end
                # Commit the scalar to the target dimension a_i once per inner loop
                out_i[$(indices[i])] += val_1 * p_2
            end
        end

        # --- 2. WRAP IN OUTER LOOPS (d = 2 up to N) ---
        for d in 2:N
            var = indices[d]
            p_curr = Symbol("p_", d)
            p_prev = Symbol("p_", d+1)

            if d == i
                # Target dimension: skip multiplying by pi[i]
                loop_expr = quote
                    for $var in axes(U_i, $d)
                        $p_curr = $p_prev
                        $loop_expr
                    end
                end
            else
                # Other dimensions: multiply running probability by pi[d]
                loop_expr = quote
                    for $var in axes(U_i, $d)
                        $p_curr = $p_prev * pi[$d][$var]
                        $loop_expr
                    end
                end
            end
        end

        # --- 3. PUSH BLOCK FOR PLAYER i ---
        push!(exprs, quote
            out_i = out[$i]
            U_i = U[$i]
            fill!(out_i, zero(eltype(out_i)))

            # p_{N+1} initializes the probability chain at 1.0
            $(Symbol("p_", N+1)) = one(eltype(out_i))

            @inbounds $loop_expr
        end)
    end

    push!(exprs, :(return out))
    return Expr(:block, exprs...)
end

function jacobian_t!(J, ubar, mu, u)
    idx = 1
    @inbounds for p in eachindex(u)
        for a in eachindex(mu[p])
            J[idx] = ubar[p][end] - ubar[p][a]
            idx += 1
        end
    end
    J
end

function residual!(out, mu, ubar, x, lambda, u)
    idx = 1
    @inbounds for p in eachindex(u)
        for a in eachindex(mu[p])
            out[idx] = mu[p][a] - lambda*(ubar[p][a] - ubar[p][end])
            idx += 1
        end
    end
    out
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
                    c = 0.0
                    for pd_a in eachindex(pi[pd_p])
                        gm = (dudpi[eq_p][pd_p][eq_a, pd_a] - dudpi[eq_p][pd_p][end, pd_a])
                        c += gm * pi[pd_p][pd_a]
                    end

                    for pd_a in 1:(size(u[pd_p], pd_p)-1)
                        gm = (dudpi[eq_p][pd_p][eq_a, pd_a] - dudpi[eq_p][pd_p][end, pd_a])
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

function uniform_xprofile(Us)
    nx = sum(size(Us[i], i) - 1 for i in eachindex(Us))
    zeros(nx)
end
