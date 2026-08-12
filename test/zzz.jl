quote
    begin
        res_pq = (result[1])[2]
        pay_p = payoffs[1]
        fill!(res_pq, zero(eltype(res_pq)))
        for a3 = 1:size(pay_p, 3)
            p3 = 1 * (pi[3])[a3]
            for a2 = 1:size(pay_p, 2)
                p2 = p3
                @simd for a1 = 1:size(pay_p, 1)
                    @inbounds res_pq[a1, a2] += pay_p[a1, a2, a3] * p2
                end
            end
        end
    end
    begin
        res_pq = (result[1])[3]
        pay_p = payoffs[1]
        fill!(res_pq, zero(eltype(res_pq)))
        for a3 = 1:size(pay_p, 3)
            p3 = 1
            for a2 = 1:size(pay_p, 2)
                p2 = p3 * (pi[2])[a2]
                @simd for a1 = 1:size(pay_p, 1)
                    @inbounds res_pq[a1, a3] += pay_p[a1, a2, a3] * p2
                end
            end
        end
    end
    begin
        res_pq = (result[2])[1]
        pay_p = payoffs[2]
        fill!(res_pq, zero(eltype(res_pq)))
        for a3 = 1:size(pay_p, 3)
            p3 = 1 * (pi[3])[a3]
            for a2 = 1:size(pay_p, 2)
                p2 = p3
                @simd for a1 = 1:size(pay_p, 1)
                    @inbounds res_pq[a2, a1] += pay_p[a1, a2, a3] * p2
                end
            end
        end
    end
    begin
        res_pq = (result[2])[3]
        pay_p = payoffs[2]
        fill!(res_pq, zero(eltype(res_pq)))
        for a3 = 1:size(pay_p, 3)
            p3 = 1
            for a2 = 1:size(pay_p, 2)
                p2 = p3
                begin
                    s = zero(eltype(res_pq))
                    @simd for a1 = 1:size(pay_p, 1)
                        @inbounds s += pay_p[a1, a2, a3] * (p2 * (pi[1])[a1])
                    end
                    @inbounds res_pq[a2, a3] += s
                end
            end
        end
    end
    begin
        res_pq = (result[3])[1]
        pay_p = payoffs[3]
        fill!(res_pq, zero(eltype(res_pq)))
        for a3 = 1:size(pay_p, 3)
            p3 = 1
            for a2 = 1:size(pay_p, 2)
                p2 = p3 * (pi[2])[a2]
                @simd for a1 = 1:size(pay_p, 1)
                    @inbounds res_pq[a3, a1] += pay_p[a1, a2, a3] * p2
                end
            end
        end
    end
    begin
        res_pq = (result[3])[2]
        pay_p = payoffs[3]
        fill!(res_pq, zero(eltype(res_pq)))
        for a3 = 1:size(pay_p, 3)
            p3 = 1
            for a2 = 1:size(pay_p, 2)
                p2 = p3
                begin
                    s = zero(eltype(res_pq))
                    @simd for a1 = 1:size(pay_p, 1)
                        @inbounds s += pay_p[a1, a2, a3] * (p2 * (pi[1])[a1])
                    end
                    @inbounds res_pq[a3, a2] += s
                end
            end
        end
    end
end
