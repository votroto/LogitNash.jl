quote
    for p = 1:N, q = 1:N
        p != q && fill!((results[p])[q], zero(T))
    end
    @inbounds begin
        begin
            pay_p = payoffs[1]
            begin
                for a2 = axes(pay_p, 2)
                    begin
                        @simd ivdep for a1 = axes(pay_p, 1)
                            val = pay_p[a1, a2]
                            ((results[1])[2])[a1, a2] += val * 1
                        end
                    end
                end
            end
        end
        begin
            pay_p = payoffs[2]
            begin
                for a2 = axes(pay_p, 2)
                    begin
                        @simd ivdep for a1 = axes(pay_p, 1)
                            val = pay_p[a1, a2]
                            ((results[2])[1])[a1, a2] += val * 1
                        end
                    end
                end
            end
        end
    end
end
