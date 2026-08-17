
function build_deriv_loops_clean(d, w, p, N)
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
            $(build_deriv_loops_clean(d - 1, next_w, p, N))
        end
    end
end

@generated function unilateral_derivatives!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    p_blocks = map(1:N) do p
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops_clean(N, init_w, p, N)
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



function unilateral_deviations_from_derivatives!(
    out::NTuple{N,Vector},
    dudpi::NTuple{N,NTuple{N,Matrix}},
    pi::NTuple{N,Vector},
) where N
    for p in 1:N
        q = p == 1 ? 2 : 1

        if p == 1
            mul!(out[p], dudpi[p][q], pi[q])
        else
            mul!(out[p], transpose(dudpi[p][q]), pi[q])
        end
    end

    return out
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
                    # Own-player identity block
                    for pd_a in 1:(size(u[eq_p], eq_p)-1)
                        J[eq_i, pd_i] = (eq_a == pd_a)
                        pd_i += 1
                    end
                else
                    d = dudpi[eq_p][pd_p]

                    # No more `transposed` logic!
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



@generated function unilateral_derivatives_fast2_cleaner!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}

    math_p_gt_1 = Expr[]
    cleanup_p_gt_1 = Expr[]

    # 1. Generate blocks for p > 1
    for p in 2:N
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops_fast2(N, init_w, p, N)

        # Pure math loop
        push!(math_p_gt_1, quote
            pay_p = payoffs[$p]
            $body
        end)

        # Cleanup routine
        push!(cleanup_p_gt_1, quote
            LinearAlgebra.transpose!(results[$p][1], results[1][$p])
            fill!(results[1][$p], zero(T))
        end)
    end

    # 2. Generate block for p = 1
    init_w_1 = Any[one(T) for _ in 1:N]
    body_1 = build_deriv_loops_fast2(N, init_w_1, 1, N)
    math_p_1 = quote
        pay_p = payoffs[1]
        $body_1
    end

    return quote
        for p in 1:N, q in 2:N
            p != q && fill!(results[p][q], zero(T))
        end

        @inbounds begin
            # PHASE 1: Keep the instruction cache hot for all p > 1 loops
            $(math_p_gt_1...)

            # PHASE 2: Do all memory transposes and workspace clearing at once
            $(cleanup_p_gt_1...)

            # PHASE 3: Run the final math block for p = 1 into the clean arrays
            $math_p_1
        end
    end
end


























function build_dev_loops(d, last_w, p, N)
    var_ad = Symbol("a", d)
    var_ap = Symbol("a", p)

    if d == 1
        actions = [Symbol("a", k) for k in 1:N]
        pay_idx = Expr(:ref, :payoff, actions...)

        # Use a scalar accumulator for SIMD when the inner loop is marginalized.
        if 1 == p
            return quote
                @simd for $var_ad in axes(payoff, $d)
                    result[$var_ap] += $pay_idx * $last_w
                end
            end
        else
            return quote
                s = zero(eltype(result))
                @simd for $var_ad in axes(payoff, $d)
                    s += $pay_idx * $last_w * pi[$d][$var_ad]
                end
                result[$var_ap] += s
            end
        end
    else
        new_w = Symbol("w", d)
        # Multiply by the strategy probability unless this is the target dimension
        w_expr = (d == p) ? :($last_w) : :($last_w * pi[$d][$var_ad])
        inner_loop = build_dev_loops(d - 1, new_w, p, N)

        return quote
            for $var_ad in axes(payoff, $d)
                $new_w = $w_expr
                $inner_loop
            end
        end
    end
end

@generated function unilateral_deviations!(
    results::NTuple{N,Vector},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    exprs = []

    for p in 1:N
        inner_loops_expr = build_dev_loops(N, 1, p, N)

        push!(exprs, quote
            result = results[$p]
            payoff = payoffs[$p]
            fill!(result, zero(eltype(result)))
            @inbounds $inner_loops_expr
        end)
    end

    push!(exprs, :(return results))
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


function uniform_xprofile(Us)
    nx = sum(size(Us[i], i) - 1 for i in eachindex(Us))
    zeros(nx)
end
