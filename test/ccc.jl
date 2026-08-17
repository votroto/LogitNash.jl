quote
    for p = 1:N, q = 2:N
        p != q && fill!((results[p])[q], zero(T))
    end
    @inbounds begin
        begin
            pay_p = payoffs[2]
            begin
                for a5 = axes(pay_p, 5)
                    w_d5_q1 = 1.0 * (pi[5])[a5]
                    w_d5_q3 = 1.0 * (pi[5])[a5]
                    w_d5_q4 = 1.0 * (pi[5])[a5]
                    begin
                        for a4 = axes(pay_p, 4)
                            w_d4_q1 = w_d5_q1 * (pi[4])[a4]
                            w_d4_q3 = w_d5_q3 * (pi[4])[a4]
                            w_d4_q5 = 1.0 * (pi[4])[a4]
                            begin
                                for a3 = axes(pay_p, 3)
                                    w_d3_q1 = w_d4_q1 * (pi[3])[a3]
                                    w_d3_q4 = w_d5_q4 * (pi[3])[a3]
                                    w_d3_q5 = w_d4_q5 * (pi[3])[a3]
                                    begin
                                        for a2 = axes(pay_p, 2)
                                            begin
                                                s_shared = zero(T)
                                                @simd ivdep for a1 = axes(pay_p, 1)
                                                    val = pay_p[a1, a2, a3, a4, a5]
                                                    ((results[1])[2])[a1, a2] += val * w_d3_q1
                                                    s_shared += val * (pi[1])[a1]
                                                end
                                                ((results[2])[3])[a2, a3] += s_shared * w_d4_q3
                                                ((results[2])[4])[a2, a4] += s_shared * w_d3_q4
                                                ((results[2])[5])[a2, a5] += s_shared * w_d3_q5
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            LinearAlgebra.transpose!((results[2])[1], (results[1])[2])
            fill!((results[1])[2], zero(T))
        end
        begin
            pay_p = payoffs[3]
            begin
                for a5 = axes(pay_p, 5)
                    w_d5_q1 = 1.0 * (pi[5])[a5]
                    w_d5_q2 = 1.0 * (pi[5])[a5]
                    w_d5_q4 = 1.0 * (pi[5])[a5]
                    begin
                        for a4 = axes(pay_p, 4)
                            w_d4_q1 = w_d5_q1 * (pi[4])[a4]
                            w_d4_q2 = w_d5_q2 * (pi[4])[a4]
                            w_d4_q5 = 1.0 * (pi[4])[a4]
                            begin
                                for a3 = axes(pay_p, 3)
                                    begin
                                        for a2 = axes(pay_p, 2)
                                            w_d2_q1 = w_d4_q1 * (pi[2])[a2]
                                            w_d2_q4 = w_d5_q4 * (pi[2])[a2]
                                            w_d2_q5 = w_d4_q5 * (pi[2])[a2]
                                            begin
                                                s_shared = zero(T)
                                                @simd ivdep for a1 = axes(pay_p, 1)
                                                    val = pay_p[a1, a2, a3, a4, a5]
                                                    ((results[1])[3])[a1, a3] += val * w_d2_q1
                                                    s_shared += val * (pi[1])[a1]
                                                end
                                                ((results[3])[2])[a3, a2] += s_shared * w_d4_q2
                                                ((results[3])[4])[a3, a4] += s_shared * w_d2_q4
                                                ((results[3])[5])[a3, a5] += s_shared * w_d2_q5
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            LinearAlgebra.transpose!((results[3])[1], (results[1])[3])
            fill!((results[1])[3], zero(T))
        end
        begin
            pay_p = payoffs[4]
            begin
                for a5 = axes(pay_p, 5)
                    w_d5_q1 = 1.0 * (pi[5])[a5]
                    w_d5_q2 = 1.0 * (pi[5])[a5]
                    w_d5_q3 = 1.0 * (pi[5])[a5]
                    begin
                        for a4 = axes(pay_p, 4)
                            begin
                                for a3 = axes(pay_p, 3)
                                    w_d3_q1 = w_d5_q1 * (pi[3])[a3]
                                    w_d3_q2 = w_d5_q2 * (pi[3])[a3]
                                    w_d3_q5 = 1.0 * (pi[3])[a3]
                                    begin
                                        for a2 = axes(pay_p, 2)
                                            w_d2_q1 = w_d3_q1 * (pi[2])[a2]
                                            w_d2_q3 = w_d5_q3 * (pi[2])[a2]
                                            w_d2_q5 = w_d3_q5 * (pi[2])[a2]
                                            begin
                                                s_shared = zero(T)
                                                @simd ivdep for a1 = axes(pay_p, 1)
                                                    val = pay_p[a1, a2, a3, a4, a5]
                                                    ((results[1])[4])[a1, a4] += val * w_d2_q1
                                                    s_shared += val * (pi[1])[a1]
                                                end
                                                ((results[4])[2])[a4, a2] += s_shared * w_d3_q2
                                                ((results[4])[3])[a4, a3] += s_shared * w_d2_q3
                                                ((results[4])[5])[a4, a5] += s_shared * w_d2_q5
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            LinearAlgebra.transpose!((results[4])[1], (results[1])[4])
            fill!((results[1])[4], zero(T))
        end
        begin
            pay_p = payoffs[5]
            begin
                for a5 = axes(pay_p, 5)
                    begin
                        for a4 = axes(pay_p, 4)
                            w_d4_q1 = 1.0 * (pi[4])[a4]
                            w_d4_q2 = 1.0 * (pi[4])[a4]
                            w_d4_q3 = 1.0 * (pi[4])[a4]
                            begin
                                for a3 = axes(pay_p, 3)
                                    w_d3_q1 = w_d4_q1 * (pi[3])[a3]
                                    w_d3_q2 = w_d4_q2 * (pi[3])[a3]
                                    w_d3_q4 = 1.0 * (pi[3])[a3]
                                    begin
                                        for a2 = axes(pay_p, 2)
                                            w_d2_q1 = w_d3_q1 * (pi[2])[a2]
                                            w_d2_q3 = w_d4_q3 * (pi[2])[a2]
                                            w_d2_q4 = w_d3_q4 * (pi[2])[a2]
                                            begin
                                                s_shared = zero(T)
                                                @simd ivdep for a1 = axes(pay_p, 1)
                                                    val = pay_p[a1, a2, a3, a4, a5]
                                                    ((results[1])[5])[a1, a5] += val * w_d2_q1
                                                    s_shared += val * (pi[1])[a1]
                                                end
                                                ((results[5])[2])[a5, a2] += s_shared * w_d3_q2
                                                ((results[5])[3])[a5, a3] += s_shared * w_d2_q3
                                                ((results[5])[4])[a5, a4] += s_shared * w_d2_q4
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            LinearAlgebra.transpose!((results[5])[1], (results[1])[5])
            fill!((results[1])[5], zero(T))
        end
        begin
            pay_p = payoffs[1]
            begin
                for a5 = axes(pay_p, 5)
                    w_d5_q2 = 1.0 * (pi[5])[a5]
                    w_d5_q3 = 1.0 * (pi[5])[a5]
                    w_d5_q4 = 1.0 * (pi[5])[a5]
                    begin
                        for a4 = axes(pay_p, 4)
                            w_d4_q2 = w_d5_q2 * (pi[4])[a4]
                            w_d4_q3 = w_d5_q3 * (pi[4])[a4]
                            w_d4_q5 = 1.0 * (pi[4])[a4]
                            begin
                                for a3 = axes(pay_p, 3)
                                    w_d3_q2 = w_d4_q2 * (pi[3])[a3]
                                    w_d3_q4 = w_d5_q4 * (pi[3])[a3]
                                    w_d3_q5 = w_d4_q5 * (pi[3])[a3]
                                    begin
                                        for a2 = axes(pay_p, 2)
                                            w_d2_q3 = w_d4_q3 * (pi[2])[a2]
                                            w_d2_q4 = w_d3_q4 * (pi[2])[a2]
                                            w_d2_q5 = w_d3_q5 * (pi[2])[a2]
                                            begin
                                                @simd ivdep for a1 = axes(pay_p, 1)
                                                    val = pay_p[a1, a2, a3, a4, a5]
                                                    ((results[1])[2])[a1, a2] += val * w_d3_q2
                                                    ((results[1])[3])[a1, a3] += val * w_d2_q3
                                                    ((results[1])[4])[a1, a4] += val * w_d2_q4
                                                    ((results[1])[5])[a1, a5] += val * w_d2_q5
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
