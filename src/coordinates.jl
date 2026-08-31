function redlograt_to_prob!(y::AbstractVector{F}, x::AbstractVector{F}, ref::Int) where F
    c = maximum(x, init=zero(F))
    denom = zero(F)

    @inbounds for i in eachindex(x)
        a = i + (i >= ref)
        v = exp(x[i] - c)
        y[a] = v
        denom += v
    end

    y[ref] = exp(-c)
    denom += y[ref]

    @inbounds for i in eachindex(y)
        y[i] = y[i] / denom
    end

    return y
end

function splitviews(x::AbstractVector, js::NTuple{N,Int}) where {N}
    offs = cumsum((0, js...))
    ntuple(i -> @view(x[(offs[i]+1):offs[i+1]]), N)
end

function extract_strategy_profiles!(pi_buffs::NTuple{N}, x, refs) where N
    mu = splitviews(x, ntuple(p -> length(pi_buffs[p]) - 1, N))

    for p in 1:N
        redlograt_to_prob!(pi_buffs[p], mu[p], refs[p])
    end

    return mu, pi_buffs
end

function uniform_xprofile(Us)
    nx = sum(size(Us[i], i) - 1 for i in eachindex(Us))
    zeros(nx)
end

function pivot_reference!(x_p::AbstractVector{Float64}, dx_p::AbstractVector{Float64}, dx_old_p::AbstractVector{Float64}, p::Int, new_ref_idx::Int, refs)
    old_ref = refs[p]
    new_ref = new_ref_idx + (new_ref_idx >= old_ref)

    val_x = x_p[new_ref_idx]
    val_dx = dx_p[new_ref_idx]
    val_dx_old = dx_old_p[new_ref_idx]

    # 1. Shift relative to the new reference
    @inbounds @simd for i in eachindex(x_p)
        x_p[i] -= val_x
        dx_p[i] -= val_dx
        dx_old_p[i] -= val_dx_old
    end

    # 2. Determine where the old reference lands in the reduced array
    dest_idx = new_ref < old_ref ? (old_ref - 1) : old_ref

    # 3. Shift elements and insert the old reference (negated)
    shift_and_insert!(x_p, new_ref_idx, dest_idx, -val_x)
    shift_and_insert!(dx_p, new_ref_idx, dest_idx, -val_dx)
    shift_and_insert!(dx_old_p, new_ref_idx, dest_idx, -val_dx_old)

    refs[p] = new_ref
end

function pivot_references!(x, dx, dx_old, pi::NTuple{N}, refs) where N
    mu_dim = ntuple(p -> length(pi[p]) - 1, N)

    mu_x = splitviews(x, mu_dim)
    mu_dx = splitviews(dx, mu_dim)
    mu_dx_old = splitviews(dx_old, mu_dim)

    for p in 1:N
        best_a = argmax(pi[p])
        if best_a != refs[p]
            idx = best_a - (best_a > refs[p])
            pivot_reference!(mu_x[p], mu_dx[p], mu_dx_old[p], p, idx, refs)
        end
    end
end