function redlograt_to_prob!(y::AbstractVector{F}, x::AbstractVector{F}, ref::Int) where F
    c = maximum(x, init=zero(F))

    nrm = exp(-c)
    @inbounds y[ref] = nrm
    @inbounds for i in eachindex(x)
        a = i + (i >= ref)
        v = exp(x[i] - c)
        y[a] = v
        nrm += v
    end

    rnrm = inv(nrm)
    @. y *= rnrm

    return y
end

function splitviews(x::AbstractVector, js::NTuple{N,Int}) where {N}
    offs = cumsum((0, js...))
    ntuple(i -> @view(x[(offs[i]+1):offs[i+1]]), Val(N))
end

function extract_strategy_profiles!(pi_buffs::NTuple{N}, x, refs) where N
    mu = splitviews(x, ntuple(p -> length(pi_buffs[p]) - 1, Val(N)))

    for p in 1:N
        redlograt_to_prob!(pi_buffs[p], mu[p], refs[p])
    end

    return mu, pi_buffs
end

function uniform_xprofile(dims)
    nx = sum(d - 1 for d in dims)
    zeros(nx)
end

function normalize_tangent_mu_t!(dmu, dt)
    nrm2 = dt^2 + squared_norm(dmu)
    rnrm = 1 / sqrt(nrm2)

    @. dmu *= rnrm
    dt *= rnrm

    return dmu, dt
end

function pivot_reference!(vs, best_idx, refs_idx)
    b = vs[best_idx]
    @. vs -= b

    step = ifelse(best_idx < refs_idx, 1, -1)
    @inbounds for i in best_idx:step:(refs_idx-step)
        vs[i] = vs[i+step]
    end
    @inbounds vs[refs_idx] = -b

    return vs
end

function pivot_references!(x, dx, dt, pi::NTuple{N}, refs) where N
    mu_dim = ntuple(p -> length(pi[p]) - 1, Val(N))

    mu = splitviews(x, mu_dim)
    dmu = splitviews(dx, mu_dim)

    changed = false
    for p in 1:N
        best_a = argmax(pi[p])
        best_idx = best_a - (best_a > refs[p])
        refs_idx = refs[p] - (best_a < refs[p])

        if best_a != refs[p] && mu[p][best_idx] >= 0.1
            changed = true
            pivot_reference!(mu[p], best_idx, refs_idx)
            pivot_reference!(dmu[p], best_idx, refs_idx)
            refs[p] = best_a
        end
    end
    if changed
        dx, dt = normalize_tangent_mu_t!(dx, dt)
    end
    changed, dx, dt
end