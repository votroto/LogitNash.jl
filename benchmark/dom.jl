@generated function is_strictly_dominated(
    U::AbstractArray{T, N},
    ::Val{p},
    a::Int,
    b::Int,
    active_actions::NTuple{N, Vector{Int}}
) where {T, N, p}

    # Build indexing expressions, e.g., U[i_1, a, i_3] vs U[i_1, b, i_3]
    idx_a = [i == p ? :a : Symbol("i_", i) for i in 1:N]
    idx_b = [i == p ? :b : Symbol("i_", i) for i in 1:N]

    # Base comparison: early exit if `a` is NOT strictly dominated by `b`
    ex = quote
        if U[$(idx_a...)] >= U[$(idx_b...)]
            return false
        end
    end

    # Wrap the expression in loops from d = 1 up to N.
    # By starting at 1 and wrapping outward, d=1 becomes the INNERMOST loop.
    for d in 1:N
        if d != p
            sym = Symbol("i_", d)
            ex = quote
                for $sym in active_actions[$d]
                    $ex
                end
            end
        end
    end

    return quote
        @inbounds $ex
        return true
    end
end

@generated function _elimination_pass!(
    payoffs::NTuple{N, AbstractArray{T, N}},
    active_actions::NTuple{N, Vector{Int}},
    to_delete::NTuple{N, Vector{Int}} # Pre-allocated buffers passed in
) where {N, T}

    exprs = Expr[]
    push!(exprs, :(any_changed = false))

    for p in 1:N
        push!(exprs, quote
            U = payoffs[$p]
            actions = active_actions[$p]
            dels = to_delete[$p]

            empty!(dels) # Zero-allocation clear
            n_acts = length(actions)

            if n_acts > 1
                for i in 1:n_acts
                    a = actions[i]
                    for j in 1:n_acts
                        i == j && continue
                        b = actions[j]

                        if is_strictly_dominated(U, Val($p), a, b, active_actions)
                            push!(dels, i) # Grows without allocating if within sizehint
                            any_changed = true
                            break
                        end
                    end
                end

                if !isempty(dels)
                    # deleteat! works in-place. dels is naturally sorted!
                    deleteat!(actions, dels)
                end
            end
        end)
    end

    push!(exprs, :(return any_changed))
    return Expr(:block, exprs...)
end

function eliminate_dominated_strategies(
    payoffs::NTuple{N, AbstractArray{T, N}};
) where {N, T}

    # 1. One-time setup allocations
    active_actions = ntuple(p -> collect(1:size(payoffs[p], p)), N)

    # Pre-allocate delete buffers. sizehint! guarantees push! will never allocate.
    to_delete = ntuple(N) do p
        buf = Int[]
        sizehint!(buf, size(payoffs[p], p))
        buf
    end

    # 2. Zero-allocation loop
    changed = true
    while changed
        changed = _elimination_pass!(payoffs, active_actions, to_delete)
    end

    # 3. Final payload extraction
    reduced_payoffs = ntuple(N) do p
        payoffs[p][active_actions...]
    end

    return reduced_payoffs, active_actions
end





