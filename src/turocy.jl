function build_deriv_loops(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        a_all = [Symbol("a", k) for k in 1:N]
        ap = Symbol("a", p)

        q_loops = map(1:N) do q
            q == p && return :()
            aq = Symbol("a", q)
            res = :(results[$p][$q])
            wq = w[q]

            if p == 1 || q == 1
                quote
                    @simd ivdep for a1 in 1:size(pay_p, 1)
                        $res[$ap, $aq] += pay_p[$(a_all...)] * $wq
                    end
                end
            else
                quote
                    s = zero(eltype($res))
                    @simd ivdep for a1 in 1:size(pay_p, 1)
                        s += pay_p[$(a_all...)] * $wq * pi[1][a1]
                    end
                    $res[$ap, $aq] += s
                end
            end
        end
        return Expr(:block, q_loops...)
    else
        assignments = Expr[]
        next_w = copy(w)

        for q in 1:N
            if q != p && d != p && d != q
                w_sym = Symbol("w_d", d, "_q", q)
                push!(assignments, :($w_sym = $(w[q]) * pi[$d][$ad]))
                next_w[q] = w_sym
            end
        end

        return quote
            for $ad in 1:size(pay_p, $d)
                $(assignments...)
                $(build_deriv_loops(d - 1, next_w, p, N))
            end
        end
    end
end

@generated function unilateral_derivatives!(
    results::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    p_blocks = map(1:N) do p
        init_w = Any[1 for _ in 1:N]
        body = build_deriv_loops(N, init_w, p, N)
        quote
            pay_p = payoffs[$p]
            $body
        end
    end

    return quote
        for p in 1:N, q in 1:N
            p != q && fill!(results[p][q], zero(eltype(results[p][q])))
        end
        @inbounds begin
            $(p_blocks...)
        end
    end
end

function unilateral_deviations_from_derivatives!(
    out::NTuple{N,Vector},
    dudpi::NTuple{N,NTuple{N,Matrix}},
    pi::NTuple{N,Vector},
) where N
    for p in 1:N
        q = p == 1 ? 2 : 1
        mul!(out[p], dudpi[p][q], pi[q])
    end

    return out
end

function build_dev_loops(d, last_w, p, N)
    var_ad = Symbol("a", d)
    var_ap = Symbol("a", p)

    if d == 1
        actions = [Symbol("a", k) for k in 1:N]
        pay_idx = Expr(:ref, :payoff, actions...)

        # Use a scalar accumulator for SIMD when the inner loop is marginalized.
        if 1 == p
            return quote
                @simd for $var_ad in axes(payoff, $d)
                    result[$var_ap] += $pay_idx * $last_w
                end
            end
        else
            return quote
                s = zero(eltype(result))
                @simd for $var_ad in axes(payoff, $d)
                    s += $pay_idx * $last_w * pi[$d][$var_ad]
                end
                result[$var_ap] += s
            end
        end
    else
        new_w = Symbol("w", d)
        # Multiply by the strategy probability unless this is the target dimension
        w_expr = (d == p) ? :($last_w) : :($last_w * pi[$d][$var_ad])
        inner_loop = build_dev_loops(d - 1, new_w, p, N)

        return quote
            for $var_ad in axes(payoff, $d)
                $new_w = $w_expr
                $inner_loop
            end
        end
    end
end

@generated function unilateral_deviations!(
    results::NTuple{N,Vector},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    exprs = []

    for p in 1:N
        inner_loops_expr = build_dev_loops(N, 1, p, N)

        push!(exprs, quote
            result = results[$p]
            payoff = payoffs[$p]
            fill!(result, zero(eltype(result)))
            @inbounds $inner_loops_expr
        end)
    end

    push!(exprs, :(return results))
    return Expr(:block, exprs...)
end

function jacobian_t!(J, ubar, mu, u)
    idx = 1
    @inbounds for p in eachindex(u)
        for a in eachindex(mu[p])
            J[idx] = ubar[p][end] - ubar[p][a]
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

function jacobian_x!(J, pi, lam, dudpi, u::NTuple{N}) where {N}
    eq_i = 1

    @inbounds for eq_p in eachindex(u)
        for eq_a in 1:(size(u[eq_p], eq_p)-1)
            pd_i = 1

            for pd_p in eachindex(u)
                if pd_p == eq_p
                    # Own-player identity block
                    for pd_a in 1:(size(u[eq_p], eq_p)-1)
                        J[eq_i, pd_i] = (eq_a == pd_a)

                        pd_i += 1
                    end
                else
                    c = 0.0
                    for pd_a in eachindex(pi[pd_p])
                        gm = (dudpi[eq_p][pd_p][eq_a, pd_a] - dudpi[eq_p][pd_p][end, pd_a])
                        c += gm * pi[pd_p][pd_a]
                    end

                    for pd_a in 1:(size(u[pd_p], pd_p)-1)
                        gm = (dudpi[eq_p][pd_p][eq_a, pd_a] - dudpi[eq_p][pd_p][end, pd_a])
                        J[eq_i, pd_i] = -lam * pi[pd_p][pd_a] * (gm - c)

                        pd_i += 1
                    end
                end
            end

            eq_i += 1
        end
    end

    return J
end

function uniform_xprofile(Us)
    nx = sum(size(Us[i], i) - 1 for i in eachindex(Us))
    zeros(nx)
end
