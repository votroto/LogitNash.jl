


function unilateral_deviations_from_derivatives!(
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






function build_deriv_loops(d, w, p, N)
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
            $(build_deriv_loops(d - 1, next_w, p, N))
        end
    end
end


@generated function unilateral_derivatives!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}

    math_p_gt_1 = Expr[]
    cleanup_p_gt_1 = Expr[]

    # 1. Generate blocks for p > 1
    for p in 2:N
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops(N, init_w, p, N)

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
    body_1 = build_deriv_loops(N, init_w_1, 1, N)
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













# Scaled the parameter column from Ft to Fλ for the log . + 1
function jacobian_t!(J, ubar, mu, u, fac)
    idx = 1
    @inbounds for p in eachindex(u)
        for a in eachindex(mu[p])
            J[idx] = fac * (ubar[p][end] - ubar[p][a])
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

# TODO: fix this ugly nonsense!

function jacobian_x!(J, pi, lam, dudpi, u::NTuple{N}) where {N}
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
