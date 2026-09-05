function unilateral_deviations_from_derivatives!(
    out::NTuple{N,Vector{Float64}},
    dudpi::NTuple{N,NTuple{N,Matrix{Float64}}},
    pi::NTuple{N,Vector{Float64}},
) where N
    for p in 1:N
        q = p == 1 ? 2 : 1
        mul!(out[p], dudpi[p][q], pi[q])
    end

    return out
end
_actsym(q) = Symbol("a", q)

function build_deriv_loops(N, p, d=N, w=Any[1.0 for _ in 1:N])
    @show w
    ad = _actsym(d)

    if d == 1
        a_all = [_actsym(k) for k in 1:N]
        ap = _actsym(p)

        writes = nothing
        accums = Expr[]

        if p == 1
            writes = [:((results[1][$q])[a1, $(_actsym(q))] += val * $(w[q])) for q in 2:N]
        else
            writes = [:((results[1][$p])[a1, $ap] += val * $(w[1]))]
            accums = [:((results[$p][$q])[$ap, $(_actsym(q))] += s_shared * $(w[q])) for q in 2:N if q != p]
        end

        return quote
            s_shared = 0.0
            @simd ivdep for a1 in axes(pay_p, 1)
                val = pay_p[$(a_all...)]
                $(writes...)
                s_shared += val * pi[1][a1]
            end
            $(accums...)
        end
    end

    # Helper to clearly define which dimensions get hoisted constants
    is_active(q) = (q != p && d != p && d != q)
    next_w = Any[is_active(q) ? Symbol("w_d", d, "_q", q) : w[q] for q in 1:N]
    assignments = [:($(next_w[q]) = $(w[q]) * pi[$d][$ad]) for q in 1:N if is_active(q)]

    return quote
        for $ad in axes(pay_p, $d)
            $(assignments...)
            $(build_deriv_loops(N, p, d - 1, next_w))
        end
    end
end

""" unilateral_derivatives!(dudpi, payoffs, pi)

Computes all the partial derivatives of U wrt π.

                       ∂Upⁱ
    dudpi[p][q][i,j] = ----
                       ∂πqʲ
"""
@generated function unilateral_derivatives!(
    results::NTuple{N,NTuple{N,Matrix{Float64}}},
    payoffs::NTuple{N,Array{R,N}},
    pi::NTuple{N,Vector{Float64}}
) where {N,R<:Real}
    nests = Expr[]
    for p in 1:N
        body = build_deriv_loops(N, p)
        push!(nests, quote
            pay_p = payoffs[$p]
            @inbounds $body
        end)
    end

    return quote
        for p in 1:N, q in 2:N
            p != q && fill!(results[p][q], 0.0)
        end

        $(nests[2:end]...)

        for p in 2:N
            LinearAlgebra.transpose!(results[p][1], results[1][p])
            fill!(results[1][p], 0.0)
        end

        $(nests[1])
    end
end





























function residual!(out, ubar, mu, lambda, refs)
    idx = 1
    @inbounds for p in eachindex(mu)
        ref_a = refs[p]
        for i in eachindex(mu[p])
            a = i + (i >= ref_a)
            out[idx] = mu[p][i] - lambda*(ubar[p][a] - ubar[p][ref_a])
            idx += 1
        end
    end
    out
end

function jacobian_t!(J, ubar, lambda, refs)
    idx = 1
    @inbounds for p in eachindex(ubar)
        ref_a = refs[p]
        for i in 1:(length(ubar[p])-1)
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

function _set_pq_block!(J::AbstractMatrix{T}, pi::NTuple{N}, lambda, dudpi, ubar, refs, p, q) where {N,T}
    dudpi_pq = dudpi[p][q]
    Rp = refs[p]
    Rq = refs[q]

    @inbounds for _j in axes(J, 2)
        j = _j + (_j >= Rq)
        lam_pi_q = -lambda * pi[q][j]

        @simd ivdep for _i in axes(J, 1)
            i = _i + (_i >= Rp)

            lhs = dudpi_pq[i, j] - dudpi_pq[Rp, j]
            rhs = ubar[p][i] - ubar[p][Rp]

            J[_i, _j] = lam_pi_q * (lhs - rhs)
        end
    end
end

function jacobian_x!(J::AbstractMatrix{T}, pi::NTuple{N}, lambda, dudpi, ubar, refs) where {N,T}
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
                _set_pq_block!(pq_block, pi, lambda, dudpi, ubar, refs, p, q)
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