function unilateral_deviations!(
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

function _build_deriv_loops(N, p, d, w)
    ad = _actsym(d)

    if d == 1
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
                val = pay_p[linear_offset+a1]
                $(writes...)
                s_shared += val * pi[1][a1]
            end
            linear_offset += size(pay_p, 1)
            $(accums...)
        end
    end

    # Define which dimensions get hoisted constants
    is_active(q) = (q != p && d != p && d != q)
    next_w = Any[is_active(q) ? Symbol("w_d", d, "_q", q) : w[q] for q in 1:N]
    assignments = [:($(next_w[q]) = $(w[q]) * pi[$d][$ad]) for q in 1:N if is_active(q)]

    return quote
        for $ad in axes(pay_p, $d)
            $(assignments...)
            $(_build_deriv_loops(N, p, d - 1, next_w))
        end
    end
end

function _build_deriv_loops(N, p)
    return quote
        pay_p = payoffs[$p]
        linear_offset = 0
        @inbounds $(_build_deriv_loops(N, p, N, Any[1.0 for _ in 1:N]))
    end
end

""" unilateral_derivatives!(dudpi, payoffs, pi)

Computes all the partial derivatives of U wrt π and writes them into `dudpi`.

    ∂Upⁱ
    ---- = dudpi[p][q][i,j]
    ∂πqʲ
"""
@generated function unilateral_derivatives!(
    results::NTuple{N,NTuple{N,Matrix{Float64}}},
    payoffs::NTuple{N,Array{R,N}},
    pi::NTuple{N,Vector{Float64}}
) where {N,R<:Real}
    nests = Expr[_build_deriv_loops(N, p) for p in 1:N]

    # Temporarily hijacks P1's dudpi for better access pattern.
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

""" Mark derivative version inside the spare diagonal results matrices. """
function unilateral_derivatives_cached!(
    results::NTuple{N,NTuple{N,Matrix{Float64}}},
    payoffs::NTuple{N,Array{R,N}},
    pi::NTuple{N,Vector{Float64}}
) where {N,R<:Real}
    is_cached = true
    for p in 1:N
        @inbounds for i in eachindex(pi[p])
            if results[p][p][i, 1] != pi[p][i]
                results[p][p][i, 1] = pi[p][i]
                is_cached = false
            end
        end
    end

    is_cached && return nothing

    unilateral_derivatives!(results, payoffs, pi)
    return nothing
end

function update_kernels!(dudpi, ubar, pi, x, refs, utils)
    extract_strategy_profiles!(pi, x, refs)

    unilateral_derivatives_cached!(dudpi, utils, pi)
    unilateral_deviations!(ubar, dudpi, pi)
end
