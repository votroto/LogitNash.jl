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

function _build_deriv_loops(N, P, d, w_dict)
    ad = _actsym(d)

    if d == 1
        if P == 1:1
            writes = Expr[]
            for q in 2:N
                push!(writes, :((results[1][$q])[a1, $(_actsym(q))] += val * $(w_dict[(1, q)])))
            end
            return quote
                @simd ivdep for a1 in axes(payoffs[1], 1)
                    val = payoffs[1][linear_offset+a1]
                    $(writes...)
                end
                linear_offset += size(payoffs[1], 1)
            end
        else
            # Fused block for p in 2:N
            inner_body = Expr[]
            push!(inner_body, :(pi1_val = pi[1][a1]))

            for p in P
                ap = _actsym(p)
                w1p = w_dict[(1, p)]
                sym_val = Symbol("val_", p)
                sym_s   = Symbol("s_shared_", p)

                push!(inner_body, :($sym_val = payoffs[$p][linear_offset+a1]))
                push!(inner_body, :((results[1][$p])[a1, $ap] += $sym_val * $w1p))
                push!(inner_body, :($sym_s += $sym_val * pi1_val))
            end

            accums = Expr[]
            for p in P
                ap = _actsym(p)
                sym_s = Symbol("s_shared_", p)
                for q in 2:N
                    if q != p
                        wpq = w_dict[(min(p, q), max(p, q))]
                        push!(accums, :((results[$p][$q])[$ap, $(_actsym(q))] += $sym_s * $wpq))
                    end
                end
            end

            inits = [:( $(Symbol("s_shared_", p)) = 0.0 ) for p in P]

            return quote
                $(inits...)
                @simd ivdep for a1 in axes(payoffs[1], 1)
                    $(inner_body...)
                end
                linear_offset += size(payoffs[1], 1)
                $(accums...)
            end
        end
    end

    # d > 1: Hoist constants dynamically for required unordered pairs (i, j)
    is_active(i, j) = (d != i && d != j)
    next_w = Dict{Tuple{Int,Int}, Any}()
    assignments = Expr[]

    for (pair, w_sym) in w_dict
        if is_active(pair[1], pair[2])
            new_sym = Symbol("w_d", d, "_", pair[1], "_", pair[2])
            push!(assignments, :($new_sym = $w_sym * pi[$d][$ad]))
            next_w[pair] = new_sym
        else
            next_w[pair] = w_sym
        end
    end

    return quote
        for $ad in axes(payoffs[1], $d)
            $(assignments...)
            $(_build_deriv_loops(N, P, d - 1, next_w))
        end
    end
end

function _build_deriv_loops(N, P)
    # Determine which unique unordered pairs (i, j) this loop block actually needs
    needed_pairs = Tuple{Int,Int}[]
    if P == 1:1
        for q in 2:N
            push!(needed_pairs, (1, q))
        end
    else
        for p in P
            push!(needed_pairs, (1, p))
            for q in 2:N
                if p != q
                    push!(needed_pairs, (min(p, q), max(p, q)))
                end
            end
        end
    end
    unique!(needed_pairs)

    # Base weight is 1.0 at d = N before being successively multiplied downstream
    w_dict = Dict{Tuple{Int,Int}, Any}(pair => 1.0 for pair in needed_pairs)

    return quote
        linear_offset = 0
        @inbounds $(_build_deriv_loops(N, P, N, w_dict))
    end
end

""" unilateral_derivatives!(dudpi, payoffs, pi)

Computes all the partial derivatives of U wrt π.

    ∂Upⁱ
    ---- = dudpi[p][q][i,j]
    ∂πqʲ
"""
@generated function unilateral_derivatives!(
    results::NTuple{N,NTuple{N,Matrix{Float64}}},
    payoffs::NTuple{N,Array{R,N}},
    pi::NTuple{N,Vector{Float64}}
) where {N,R<:Real}
    # Two fused blocks: one shared for p != 1, one distinct for p == 1
    nest_p2N = _build_deriv_loops(N, 2:N)
    nest_p1  = _build_deriv_loops(N, 1:1)

    return quote
        for p in 1:N, q in 2:N
            p != q && fill!(results[p][q], 0.0)
        end

        $nest_p2N

        for p in 2:N
            LinearAlgebra.transpose!(results[p][1], results[1][p])
            fill!(results[1][p], 0.0)
        end

        $nest_p1
    end
end

""" Mark derivative version inside the pointless diagonal results matrices. """
function unilateral_derivatives_cached!(
    results::NTuple{N,NTuple{N,Matrix{Float64}}},
    payoffs::NTuple{N,Array{R,N}},
    pi::NTuple{N,Vector{Float64}}
) where {N,R<:Real}
    is_cached = true
    for p in 1:N
        @inbounds for i in eachindex(pi[p])
            if results[p][p][i, 1] != pi[p][i]
                is_cached = false
                break
            end
        end
        is_cached || break
    end

    is_cached && return nothing

    for p in 1:N
        @inbounds for i in eachindex(pi[p])
            results[p][p][i, 1] = pi[p][i]
        end
    end

    unilateral_derivatives!(results, payoffs, pi)
    return nothing
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