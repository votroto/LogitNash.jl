using LinearAlgebra

# ==============================================================================
# PASS 1 MATH GENERATOR (Deviations only)
# Tracks a single scalar weight. Extremely lightweight.
# ==============================================================================
function build_pass1_loops(d, w_sym, p, N)
    ad = Symbol("a", d)
    target_q = p == 1 ? 2 : 1 # We just need one opponent to get deviations

    if d == 1
        a_all = [Symbol("a", k) for k in 1:N]
        ap = Symbol("a", p)
        if p == 1
            return quote
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    (results[1][2])[a1, a2] += val * $w_sym
                end
            end
        else
            return quote
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    (results[1][$p])[a1, $ap] += val * $w_sym
                end
            end
        end
    end

    # Only hoist weight if we are not at the player's or target opponent's dimension
    if (d != p) && (d != target_q)
        next_w = Symbol("w_d", d, "_p", p)
        return quote
            for $ad in axes(pay_p, $d)
                $next_w = $w_sym * pi[$d][$ad]
                $(build_pass1_loops(d - 1, next_w, p, N))
            end
        end
    else
        return quote
            for $ad in axes(pay_p, $d)
                $(build_pass1_loops(d - 1, w_sym, p, N))
            end
        end
    end
end

@generated function compute_pass1!(results, payoffs, pi::NTuple{N,Vector{T}}) where {N,T}
    p_blocks = map(1:N) do p
        body = build_pass1_loops(N, :(one(T)), p, N)
        quote
            pay_p = payoffs[$p]
            $body
        end
    end
    return quote
        @inbounds begin $(p_blocks...) end
    end
end

# ==============================================================================
# PASS 2 MATH GENERATOR (Remainder Derivatives)
# ==============================================================================
function build_pass2_loops(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        a_all = [Symbol("a", k) for k in 1:N]
        ap = Symbol("a", p)

        if p == 1
            # Skip q=2 because it was computed in Pass 1
            writes = [:( (results[1][$q])[a1, $(Symbol("a", q))] += val * $(w[q]) ) for q in 3:N]
            isempty(writes) && return quote end

            return quote
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    $(writes...)
                end
            end
        else
            # Skip q=1 because it was computed in Pass 1
            accumulators = [:( (results[$p][$q])[$ap, $(Symbol("a", q))] += s_shared * $(w[q]) ) for q in 2:N if q != p]
            isempty(accumulators) && return quote end

            return quote
                s_shared = zero(T)
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    s_shared += val * pi[1][a1]
                end
                $(accumulators...)
            end
        end
    end

    # Active q's for remainder: skip q=1 generally, and q=2 if p=1
    is_active(q) = (q != p && d != p && d != q) && (p == 1 ? q > 2 : q > 1)
    next_w = Any[is_active(q) ? Symbol("w_d", d, "_q", q) : w[q] for q in 1:N]
    assignments = [:( $(next_w[q]) = $(w[q]) * pi[$d][$ad] ) for q in 1:N if is_active(q)]

    return quote
        for $ad in axes(pay_p, $d)
            $(assignments...)
            $(build_pass2_loops(d - 1, next_w, p, N))
        end
    end
end

@generated function compute_pass2!(results, payoffs, pi::NTuple{N,Vector{T}}) where {N,T}
    N <= 2 && return :(nothing) # N=2 has no remainder!

    p_blocks = map(1:N) do p
        init_w = Any[one(T) for _ in 1:N]
        body = build_pass2_loops(N, init_w, p, N)
        quote
            pay_p = payoffs[$p]
            $body
        end
    end
    return quote
        @inbounds begin $(p_blocks...) end
    end
end

# ==============================================================================
# WRAPPERS
# ==============================================================================

function unilateral_deviations_pass!(
    devs::NTuple{N,Vector{T}},
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}

    # 1. Zero out ONLY the matrices Pass 1 touches
    fill!(results[1][2], zero(T))
    for p in 2:N
        fill!(results[1][p], zero(T))
    end

    # 2. RUN HOT MATH (Contiguous memory streaming)
    compute_pass1!(results, payoffs, pi)

    # 3. Compute deviations
    mul!(devs[1], results[1][2], pi[2])
    for p in 2:N
        mul!(devs[p], transpose(results[1][p]), pi[1])
    end

    # 4. Transpose & Clean up workspaces for Pass 2 / Next Iteration
    for p in 2:N
        transpose!(results[p][1], results[1][p])
        fill!(results[1][p], zero(T))
    end

    return devs
end


function unilateral_derivatives_remainder!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}

    N <= 2 && return results

    # 1. Zero out all the remaining target matrices
    for q in 3:N
        fill!(results[1][q], zero(T))
    end
    for p in 2:N, q in 2:N
        if p != q
            fill!(results[p][q], zero(T))
        end
    end

    # 2. RUN HOT MATH (Contiguous memory streaming)
    compute_pass2!(results, payoffs, pi)

    return results
end