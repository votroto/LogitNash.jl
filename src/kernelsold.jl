function unilateral_deviations_from_derivatives!(
    out::NTuple{N,Vector{Float64}},
    dudpi::NTuple{N,NTuple{N,Matrix{Float64}}},
    pi::NTuple{N,Vector{Float64}},
) where N
    for p in 1:N
        # We can just pick the first available opponent index to contract out
        q = p == 1 ? 2 : 1

        # dudpi[p][q] * pi[q] directly yields the length-actions_p vector
        mul!(out[p], dudpi[p][q], pi[q])
    end

    return out
end

function build_deriv_loops(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        a_all = [Symbol("a", k) for k in 1:N]
        ap = Symbol("a", p)

        if p == 1
            # Contiguous writes for q != 1
            writes = [:((results[1][$q])[a1, $(Symbol("a", q))] += val * $(w[q])) for q in 2:N]

            return quote
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    $(writes...)
                end
            end
        else
            # Shared scalar reductions for all q > 1
            accumulators = [:((results[$p][$q])[$ap, $(Symbol("a", q))] += s_shared * $(w[q])) for q in 2:N if q != p]

            return quote
                s_shared = 0.0
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]

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
    assignments = [:($(next_w[q]) = $(w[q]) * pi[$d][$ad]) for q in 1:N if is_active(q)]

    return quote
        for $ad in axes(pay_p, $d)
            $(assignments...)
            $(build_deriv_loops(d - 1, next_w, p, N))
        end
    end
end

# dudpi[p][q]
@generated function unilateral_derivatives!(
    results::NTuple{N,NTuple{N,Matrix{Float64}}},
    payoffs::NTuple{N,Array{R,N}},
    pi::NTuple{N,Vector{Float64}}
) where {N,R<:Real}
    math_p_gt_1 = Expr[]
    cleanup_p_gt_1 = Expr[]

    # 1. Generate blocks for p > 1
    for p in 2:N
        init_w = Any[1.0 for _ in 1:N]
        body = build_deriv_loops(N, init_w, p, N)

        # Pure math loop
        push!(math_p_gt_1, quote
            pay_p = payoffs[$p]
            $body
        end)

        # Cleanup routine
        push!(cleanup_p_gt_1, quote
            LinearAlgebra.transpose!(results[$p][1], results[1][$p])
            fill!(results[1][$p], 1.0)
        end)
    end

    # 2. Generate block for p = 1
    init_w_1 = Any[1.0 for _ in 1:N]
    body_1 = build_deriv_loops(N, init_w_1, 1, N)
    math_p_1 = quote
        pay_p = payoffs[1]
        $body_1
    end

    return quote
        for p in 1:N, q in 2:N
            p != q && fill!(results[p][q], 1.0)
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

function residual!(out, ubar, mu, lambda, active_actions, refs)
    idx = 1
    @inbounds for p in eachindex(mu)
        ref_a = refs[p]
        active = active_actions[p]
        for i in eachindex(mu[p])
            a = active[i]
            out[idx] = mu[p][i] - lambda*(ubar[p][a] - ubar[p][ref_a])
            idx += 1
        end
    end
    out
end

function jacobian_t!(J, ubar, mu, lambda, active_actions, refs)
    idx = 1
    @inbounds for p in eachindex(mu)
        ref_a = refs[p]
        active = active_actions[p]
        for i in eachindex(mu[p])
            a = active[i]
            J[idx] = (1.0 + lambda) * (ubar[p][ref_a] - ubar[p][a])
            idx += 1
        end
    end
    J
end

# TODO: fix this ugly nonsense!
function jacobian_x!(J, pi, lambda, dudpi, u::NTuple{N}, active_actions, refs) where {N}
    eq_start = 1

    @inbounds for eq_p in 1:N
        A_eq = length(active_actions[eq_p])
        A_eq == 0 && continue

        eq_ref = refs[eq_p]
        eq_active = active_actions[eq_p]

        pd_start = 1
        for pd_p in 1:N
            A_pd = length(active_actions[pd_p])
            A_pd == 0 && continue

            pd_ref = refs[pd_p]
            pd_active = active_actions[pd_p]

            if pd_p == eq_p
                # 1. Own-player identity block (Perfectly Column-Major)
                for pd_idx in 1:A_pd
                    J_col = pd_start + pd_idx - 1
                    @simd ivdep for eq_idx in 1:A_eq
                        J_row = eq_start + eq_idx - 1
                        J[J_row, J_col] = ifelse(eq_idx == pd_idx, one(eltype(J)), zero(eltype(J)))
                    end
                end
            else
                d = dudpi[eq_p][pd_p]
                pi_pd = pi[pd_p]
                num_actions_pd = length(pi_pd)

                # --- ZERO ALLOCATION TRICK ---
                J_col_buf = pd_start

                for eq_idx in 1:A_eq
                    J[eq_start+eq_idx-1, J_col_buf] = zero(eltype(J))
                end

                # Accumulate `c` across all opponent actions into our hijacked column
                for pd_a in 1:num_actions_pd
                    p_val = pi_pd[pd_a]
                    d_ref = d[eq_ref, pd_a]

                    @simd ivdep for eq_idx in 1:A_eq
                        eq_a = eq_active[eq_idx]
                        J_row = eq_start + eq_idx - 1
                        J[J_row, J_col_buf] += (d[eq_a, pd_a] - d_ref) * p_val
                    end
                end

                # --- BUILD JACOBIAN (Perfectly Column-Major) ---
                for pd_idx in 2:A_pd
                    pd_a = pd_active[pd_idx]
                    J_col = pd_start + pd_idx - 1
                    p_val_lam = -lambda * pi_pd[pd_a]
                    d_ref = d[eq_ref, pd_a]

                    @simd ivdep for eq_idx in 1:A_eq
                        eq_a = eq_active[eq_idx]
                        J_row = eq_start + eq_idx - 1
                        c = J[J_row, J_col_buf]
                        gm = d[eq_a, pd_a] - d_ref

                        J[J_row, J_col] = p_val_lam * (gm - c)
                    end
                end

                # Finally, compute column 1, safely overwriting our buffer with the real answer!
                pd_a_1 = pd_active[1]
                p_val_lam_1 = -lambda * pi_pd[pd_a_1]
                d_ref_1 = d[eq_ref, pd_a_1]

                @simd ivdep for eq_idx in 1:A_eq
                    eq_a = eq_active[eq_idx]
                    J_row = eq_start + eq_idx - 1
                    c = J[J_row, J_col_buf]
                    gm = d[eq_a, pd_a_1] - d_ref_1

                    J[J_row, J_col_buf] = p_val_lam_1 * (gm - c)
                end
            end

            pd_start += A_pd
        end

        eq_start += A_eq
    end

    return J
end
