
function pseudo_arclength(mu, t, guess_mu, guess_t, tangent_mu, tangent_t)
    len = (t - guess_t) * tangent_t
    @inbounds @simd for j in eachindex(mu)
        len += (mu[j] - guess_mu[j]) * tangent_mu[j]
    end
    len
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

function system_jac_Ft!(J, ubar, lambda, refs)
    idx = 1
    @inbounds for p in eachindex(ubar)
        ref_a = refs[p]
        for i in 1:(length(ubar[p])-1)
            a = i + (i >= ref_a)
            J[idx, end] = (1.0 + lambda) * (ubar[p][ref_a] - ubar[p][a])
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

function system_jac_Fx!(J::AbstractMatrix{T}, pi::NTuple{N}, lambda, dudpi, ubar, refs) where {N,T}
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

function system_jac_dx_dt!(J, dx, dt)
    @inbounds for j in eachindex(dx)
        J[end, j] = dx[j]
    end
    @inbounds J[end, end] = dt
end

function system_rhs_F_g!(rhs_aug::Vector{R}, residual::Vector{R}, arc::R) where R
    @inbounds for j in eachindex(residual)
        rhs_aug[j] = -residual[j]
    end
    @inbounds rhs_aug[end] = -arc
end

function system_rhs_0_1!(rhs_aug::Vector{R}) where R
    fill!(rhs_aug, zero(R))
    @inbounds rhs_aug[end] = one(R)
end
