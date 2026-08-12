The input is a multiplayer strategic game defined by utility tensors $(u_p)_{p \in N}$.

The probability that player $q$ plays action $j$ is denoted by $\pi_q^j$.

The payoff of player $p$ when deviating to action $i$ is $U_p^i$.

The unilateral deviation utility of player $p$ to action $i$ is

$$
U_p^i(\pi_{-p})
=
\sum_{a_{-p}} u_p(i,a_{-p}) \prod_{q\neq p}\pi_q^{a_q}
$$

The derivative with respect to the probability that player $q$ plays action $j$ is

$$
\frac{\partial U_p^i}{\partial \pi_q^j}
=
\mathbf 1_{{p\neq q}} \sum_{a_{-(p,q)}} u_p(i,j,a_{-(p,q)}) \prod_{r\neq p,q}\pi_r^{a_r}
$$





Both are executed repeatedly in a predictor-corrector continuation loop about 100 times.

In the predictor I always need both, so i construct the deviations from the derivatives.
In the corrector there is an early exit path when the residual is good enough, and i only need the deviations for that, so i compute them first and the derivatives only if the jacobian is needed for the update.

However the code is quite complex. It strikes me that permuting the dimensions so that at least one of the marginalized players is contiguous in memory could improve the code readability and lower cyclomatic complexity by making the computations more symmetric between players, while simultaneously being potentially faster. am I right?


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
                quote
                    @inbounds res_pq[$var_ap, $var_aq] += $pay_idx
                end
            else
                quote
                    @inbounds res_pq[$var_ap, $var_aq] += $pay_idx * $prev_p
                end
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

@generated function unilateral_deviations!(
    out::NTuple{N,Vector},
    U::NTuple{N,Array{<:Real,N}},
    pi::NTuple{N,Vector}
) where {N}
    exprs = []

    for i in 1:N
        # Symbols for indices: a_1, a_2, ..., a_N
        indices = [Symbol("a_", k) for k in 1:N]

        # --- 1. INNERMOST LOOP (d = 1) ---
        if i == 1
            # Player 1: a_1 is the target dimension, accumulate directly into out_1
            loop_expr = quote
                @simd for a_1 in axes(U_i, 1)
                    out_i[a_1] += U_i[$(indices...)] * p_2
                end
            end
        else
            # Players i > 1: a_1 is NOT the target. Accumulate into a scalar register first!
            loop_expr = quote
                val_1 = zero(eltype(out_i))
                @simd for a_1 in axes(U_i, 1)
                    val_1 += U_i[$(indices...)] * pi[1][a_1]
                end
                # Commit the scalar to the target dimension a_i once per inner loop
                out_i[$(indices[i])] += val_1 * p_2
            end
        end

        # --- 2. WRAP IN OUTER LOOPS (d = 2 up to N) ---
        for d in 2:N
            var = indices[d]
            p_curr = Symbol("p_", d)
            p_prev = Symbol("p_", d+1)

            if d == i
                # Target dimension: skip multiplying by pi[i]
                loop_expr = quote
                    for $var in axes(U_i, $d)
                        $p_curr = $p_prev
                        $loop_expr
                    end
                end
            else
                # Other dimensions: multiply running probability by pi[d]
                loop_expr = quote
                    for $var in axes(U_i, $d)
                        $p_curr = $p_prev * pi[$d][$var]
                        $loop_expr
                    end
                end
            end
        end

        # --- 3. PUSH BLOCK FOR PLAYER i ---
        push!(exprs, quote
            out_i = out[$i]
            U_i = U[$i]
            fill!(out_i, zero(eltype(out_i)))

            # p_{N+1} initializes the probability chain at 1.0
            $(Symbol("p_", N+1)) = one(eltype(out_i))

            @inbounds $loop_expr
        end)
    end

    push!(exprs, :(return out))
    return Expr(:block, exprs...)
end