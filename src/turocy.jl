function build_deriv_loops(dims, idx, prev_p, p, q, N)
    d = dims[idx]
    var_ad = Symbol("a", d)
    var_ap = Symbol("a", p)
    var_aq = Symbol("a", q)

    if idx == length(dims)
        # --- INNERMOST LOOP (always d = 1) ---
        u_args = [Symbol("a", k) for k in 1:N]
        pay_idx = Expr(:ref, :pay_p, u_args...)

        if d == p || d == q
            # Case A: Innermost loop varies an output index.
            # Safe to write directly to res_pq.
            body = if prev_p === nothing
                quote @inbounds res_pq[$var_ap, $var_aq] += $pay_idx end
            else
                quote @inbounds res_pq[$var_ap, $var_aq] += $pay_idx * $prev_p end
            end

            return quote
                @simd for $var_ad in 1:size(pay_p, $d)
                    $body
                end
            end
        else
            # Case B: Innermost loop is a marginalized dimension.
            # Output indices are constant. Use a scalar accumulator for SIMD!
            p_term = prev_p === nothing ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])

            return quote
                s = zero(eltype(res_pq))
                @simd for $var_ad in 1:size(pay_p, $d)
                    @inbounds s += $pay_idx * $p_term
                end
                @inbounds res_pq[$var_ap, $var_aq] += s
            end
        end
    else
        # --- OUTER LOOPS ---
        if d == p || d == q
            # Skip probability multiplication for output dimensions
            inner_loop = build_deriv_loops(dims, idx + 1, prev_p, p, q, N)

            return quote
                for $var_ad in 1:size(pay_p, $d)
                    $inner_loop
                end
            end
        else
            # Multiply probability for marginalized dimensions
            new_p = Symbol("p", d)
            p_expr = prev_p === nothing ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])
            inner_loop = build_deriv_loops(dims, idx + 1, new_p, p, q, N)

            return quote
                for $var_ad in 1:size(pay_p, $d)
                    $new_p = $p_expr
                    $inner_loop
                end
            end
        end
    end
end

@generated function unilateral_derivatives!(
    result::NTuple{N,NTuple{N,Matrix}},
    payoffs::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where N
    exprs = []

    for p in 1:N
        for q in 1:N
            p == q && continue
            dims = N:-1:1
            inner_loops_expr = build_deriv_loops(dims, 1, nothing, p, q, N)

            push!(exprs, quote
                res_pq = result[$p][$q]
                pay_p = payoffs[$p]
                fill!(res_pq, zero(eltype(res_pq)))
                $inner_loops_expr
            end)
        end
    end

    return Expr(:block, exprs...)
end

function build_deviation_loops(dims, idx, prev_p, N, i)
    d = dims[idx]
    var_ad = Symbol("a", d)

    if idx == length(dims)
        # --- INNERMOST LOOP (always a1) ---
        u_idx = Expr(:ref, :U_i, [Symbol("a", k) for k in 1:N]...)

        if d == i
            # Player 1: a1 is innermost, accumulate directly into out_i[a1]
            body = quote out_i[$var_ad] += $u_idx * $prev_p end
        else
            # Players 2-N: accumulate into scalar s
            p_term = prev_p == :one ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])
            body = quote s += $u_idx * $p_term end
        end

        return quote
            @simd for $var_ad in 1:size(U_i, $d)
                $body
            end
        end
    else
        # --- OUTER LOOPS ---
        if d == i
            inner_loop = build_deviation_loops(dims, idx + 1, prev_p, N, i)
            return quote
                for $var_ad in 1:size(U_i, $d)
                    s = zero(eltype(out_i))
                    $inner_loop
                    out_i[$var_ad] += s
                end
            end
        else
            new_p = Symbol("p", d)
            p_expr = prev_p == :one ? :(pi[$d][$var_ad]) : :($prev_p * pi[$d][$var_ad])
            inner_loop = build_deviation_loops(dims, idx + 1, new_p, N, i)

            return quote
                for $var_ad in 1:size(U_i, $d)
                    $new_p = $p_expr
                    $inner_loop
                end
            end
        end
    end
end

@generated function unilateral_deviations!(
    out::NTuple{N,Vector},
    U::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where {N}
    exprs = []
    dims = N:-1:1

    for i in 1:N
        inner_loops_expr = build_deviation_loops(dims, 1, :one, N, i)

        push!(exprs, quote
            out_i = out[$i]
            U_i = U[$i]

            fill!(out_i, zero(eltype(out_i)))

            @inbounds begin
                $inner_loops_expr
            end
        end)
    end

    return Expr(:block, exprs...)
end

function jacobian_t!(J, ubar, mu, u)
    idx = 1
    for p in eachindex(u)
        for a in eachindex(mu[p])
            J[idx] = ubar[p][end] - ubar[p][a]
            idx += 1
        end
    end
    J
end

function residual!(out, mu, ubar, x, lambda, u)
    idx = 1
    for p in eachindex(u)
        for a in eachindex(mu[p])
            out[idx] = mu[p][a] - lambda*(ubar[p][a] - ubar[p][end])
            idx += 1
        end
    end
    out
end

function jacobian_x!(J, pi, lam, dudpi, u::NTuple{N}) where {N}
    eq_i = 1

    for eq_p in eachindex(u)
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
