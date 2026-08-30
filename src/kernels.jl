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

function residual!(out, ubar, mu, lambda, refs)
    idx = 1
    @inbounds for p in eachindex(mu)
        ref_a = refs[p]
        for i in eachindex(mu[p])
            # Branchless: action index leaps over the reference action
            a = i + (i >= ref_a)
            out[idx] = mu[p][i] - lambda*(ubar[p][a] - ubar[p][ref_a])
            idx += 1
        end
    end
    out
end

function jacobian_t!(J, ubar, mu, lambda, refs)
    idx = 1
    @inbounds for p in eachindex(mu)
        ref_a = refs[p]
        for i in eachindex(mu[p])
            a = i + (i >= ref_a)
            J[idx] = (1.0 + lambda) * (ubar[p][ref_a] - ubar[p][a])
            idx += 1
        end
    end
    J
end

function _set_I_block!(J::AbstractMatrix{T}) where {T}
    @inbounds for j in axes(J, 1)
        @simd ivdep for i in axes(J, 2)
            J[i, j] = ifelse(i == j, one(T), zero(T))
        end
    end
end

function _set_pq_block!(J::AbstractMatrix{T}, pi::NTuple{N}, lambda, dudpi, refs, p, q) where {N,T}
    dudpi_pq = dudpi[p][q]
    Rp = refs[p]
    Rq = refs[q]

    # Precalculate RHS sum into the FIRST column.
    @inbounds for _i in axes(J, 1)
        i = _i + (_i >= Rp)

        rhs_val = zero(T)
        @simd ivdep for k in axes(dudpi_pq, 2)
            rhs_val += (dudpi_pq[i, k] - dudpi_pq[Rp, k]) * pi[q][k]
        end
        J[_i, 1] = rhs_val
    end

    @inbounds for _j in size(J, 2):-1:1
        j = _j + (_j >= Rq)
        lam_pi_q = -lambda * pi[q][j]

        @simd ivdep for _i in axes(J, 1)
            i = _i + (_i >= Rp)

            lhs = dudpi_pq[i, j] - dudpi_pq[Rp, j]
            rhs = J[_i, 1]

            J[_i, _j] = lam_pi_q * (lhs - rhs)
        end
    end
end

function jacobian_x!(J::AbstractMatrix{T}, pi::NTuple{N}, lambda, dudpi, refs) where {N,T}
    eq_start = 1

    @inbounds for p in 1:N
        Dp = length(pi[p]) - 1

        pd_start = 1
        for q in 1:N
            Dq = length(pi[q]) - 1

            pq_block = @view J[eq_start:(eq_start+Dp-1), pd_start:(pd_start+Dq-1)]

            if q == p
                _set_I_block!(pq_block)
            else
                _set_pq_block!(pq_block, pi, lambda, dudpi, refs, p, q)
            end

            pd_start += Dq
        end
        eq_start += Dp
    end

    return J
end

function jacobian_aug!(J, dx, dt)
    @inbounds for j in eachindex(dx)
        J[end, j] = dx[j]
    end
    J[end, end] = dt
end