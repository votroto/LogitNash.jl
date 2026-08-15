
function build_deriv_loops_old(d, w, p, N)
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
                    @simd ivdep for a1 in axes(pay_p, 1)
                        $res[$ap, $aq] += pay_p[$(a_all...)] * $wq
                    end
                end
            else
                quote
                    s = zero(T)
                    @simd ivdep for a1 in axes(pay_p, 1)
                        s += pay_p[$(a_all...)] * pi[1][a1]
                    end
                    $res[$ap, $aq] += s * $wq
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
            for $ad in axes(pay_p, $d)
                $(assignments...)
                $(build_deriv_loops_old(d - 1, next_w, p, N))
            end
        end
    end
end

@generated function unilateral_derivatives_old!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    p_blocks = map(1:N) do p
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops_old(N, init_w, p, N)
        quote
            pay_p = payoffs[$p]
            $body
        end
    end

    return quote
        for p in 1:N, q in 1:N
            p != q && fill!(results[p][q], zero(T))
        end
         @inbounds begin
            $(p_blocks...)
        end
    end
end






function build_deriv_loops_clean(d, w, p, N)
    ad = Symbol("a", d)

    if d == 1
        a_all = [Symbol("a", k) for k in 1:N]
        ap = Symbol("a", p)

        if p == 1
            # Contiguous writes for q != 1
            writes = [:( (results[1][$q])[a1, $(Symbol("a", q))] += val * $(w[q]) ) for q in 2:N]

            return quote
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    $(writes...)
                end
            end
        else
            # Shared scalar reductions for all q > 1
            accumulators = [:( (results[$p][$q])[$ap, $(Symbol("a", q))] += s_shared * $(w[q]) ) for q in 2:N if q != p]

            return quote
                s_shared = zero(T)
                @simd ivdep for a1 in axes(pay_p, 1)
                    val = pay_p[$(a_all...)]
                    (results[$p][1])[a1, $ap] += val * $(w[1])
                    s_shared += val * pi[1][a1]
                end
                $(accumulators...)
            end
        end
    end

    # Helper to clearly define which dimensions get hoisted constants
    is_active(q) = (q != p && d != p && d != q)
    next_w = Any[is_active(q) ? Symbol("w_d", d, "_q", q) : w[q] for q in 1:N]
    assignments = [:( $(next_w[q]) = $(w[q]) * pi[$d][$ad] ) for q in 1:N if is_active(q)]

    return quote
        for $ad in axes(pay_p, $d)
            $(assignments...)
            $(build_deriv_loops_clean(d - 1, next_w, p, N))
        end
    end
end

@generated function unilateral_derivatives_fast!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    p_blocks = map(1:N) do p
        init_w = Any[one(T) for _ in 1:N]
        body = build_deriv_loops_clean(N, init_w, p, N)
        quote
            pay_p = payoffs[$p]
            $body
        end
    end

    return quote
        for p in 1:N, q in 1:N
            p != q && fill!(results[p][q], zero(T))
        end
        @inbounds begin
            $(p_blocks...)
        end
    end
end



function unilateral_derivatives_fast_body!(
    results::NTuple{N,NTuple{N,Matrix{T}}},
    payoffs::NTuple{N,Array{T,N}},
    pi::NTuple{N,Vector{T}}
) where {N,T}
    p_blocks = map(1:N) do p
        init_w = Any[1 for _ in 1:N]
        body = build_deriv_loops_clean(N, init_w, p, N)
        quote
            pay_p = payoffs[$p]
            $body
        end
    end

    return quote
        for p in 1:N, q in 1:N
            p != q && fill!(results[p][q], zero(T))
        end
        @inbounds begin
            $(p_blocks...)
        end
    end
end






using LinearAlgebra




using BenchmarkTools



A = 150
D = 3
dims = ntuple(_ -> A, D);
Us = ntuple(_ -> randn(dims...), D);

pi = ntuple(_ -> normalize(rand(A),1), D);

dudpi1 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D);
dudpi2 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D);


D = length(dims) # Or N
dudpi1_transposed = ntuple(p -> ntuple(q ->
    (p != 1 && q == 1) ? zeros(dims[q], dims[p]) : zeros(dims[p], dims[q]),
D), D)


unilateral_derivatives_old!(dudpi1, Us, pi);
unilateral_derivatives_fast!(dudpi1_transposed, Us, pi);

@time dudpi2 = ntuple(p -> ntuple(q ->
    (p != 1 && q == 1) ? collect(transpose(dudpi1_transposed[p][q])) : dudpi1_transposed[p][q],
D), D)

@show ([ norm.(dudpi1[i] .- dudpi2[i]) for i in eachindex(dudpi1)])



a = @benchmark unilateral_derivatives_old!($dudpi1, $Us, $pi)
b = @benchmark unilateral_derivatives_fast!($dudpi1_transposed, $Us, $pi)
