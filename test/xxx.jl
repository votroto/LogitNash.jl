quote
    begin
        begin
            res_p1_q2 = (result[1])[2]
            fill!(res_p1_q2, zero(eltype(res_p1_q2)))
        end
        begin
            res_p1_q3 = (result[1])[3]
            fill!(res_p1_q3, zero(eltype(res_p1_q3)))
        end
        pay_p = payoffs[1]
        begin
            for a3 = 1:size(pay_p, 3)
                p_d3_q2 = @inbounds((pi[3])[a3])
                begin
                    for a2 = 1:size(pay_p, 2)
                        p_d2_q3 = @inbounds((pi[2])[a2])
                        begin
                            @simd for a1 = 1:size(pay_p, 1)
                                val = @inbounds(pay_p[a1, a2, a3])
                                @inbounds res_p1_q2[a1, a2] += val * p_d3_q2
                                @inbounds res_p1_q3[a1, a3] += val * p_d2_q3
                            end
                        end
                    end
                end
            end
        end
    end
    begin
        begin
            res_p2_q1 = (result[2])[1]
            fill!(res_p2_q1, zero(eltype(res_p2_q1)))
        end
        begin
            res_p2_q3 = (result[2])[3]
            fill!(res_p2_q3, zero(eltype(res_p2_q3)))
        end
        pay_p = payoffs[2]
        begin
            for a3 = 1:size(pay_p, 3)
                p_d3_q1 = @inbounds((pi[3])[a3])
                begin
                    for a2 = 1:size(pay_p, 2)
                        begin
                            s_q3 = zero(eltype(res_p2_q3))
                            @simd for a1 = 1:size(pay_p, 1)
                                val = @inbounds(pay_p[a1, a2, a3])
                                @inbounds res_p2_q1[a2, a1] += val * p_d3_q1
                                s_q3 += val * @inbounds((pi[1])[a1])
                            end
                            @inbounds res_p2_q3[a2, a3] += s_q3
                        end
                    end
                end
            end
        end
    end
    begin
        begin
            res_p3_q1 = (result[3])[1]
            fill!(res_p3_q1, zero(eltype(res_p3_q1)))
        end
        begin
            res_p3_q2 = (result[3])[2]
            fill!(res_p3_q2, zero(eltype(res_p3_q2)))
        end
        pay_p = payoffs[3]
        begin
            for a3 = 1:size(pay_p, 3)
                begin
                    for a2 = 1:size(pay_p, 2)
                        p_d2_q1 = @inbounds((pi[2])[a2])
                        begin
                            s_q2 = zero(eltype(res_p3_q2))
                            @simd for a1 = 1:size(pay_p, 1)
                                val = @inbounds(pay_p[a1, a2, a3])
                                @inbounds res_p3_q1[a3, a1] += val * p_d2_q1
                                s_q2 += val * @inbounds((pi[1])[a1])
                            end
                            @inbounds res_p3_q2[a3, a2] += s_q2
                        end
                    end
                end
            end
        end
    end
end
