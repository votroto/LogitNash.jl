using LinearAlgebra
using Polyester

# Mock work, just to make a concrete example, the actual work is 100s of loc.
function kernel_block!(ys, A, xs, i)
    mul!(ys[i], A, xs[i])
end

function kernel!(ys::NTuple{N}, A, xs::NTuple{N}) where N
    Threads.@threads :static for i in 1:N
        kernel_block!(ys, A, xs, i)
    end
end

function outer_loop(ys, A, xs)
    for i in 1:1000
        kernel!(ys, A, xs)

        # Mock outerloop work, just to enforce iteration dependence.
        for i in eachindex(ys)
            copyto!(xs[i], ys[i])
        end
    end
end

sa = 100
sn = 10

A = randn(sa, sa)
ys = ntuple(_ -> rand(sa), sn)
xs = ntuple(_ -> rand(sa), sn)

outer_loop(ys, A, xs)

@benchmark outer_loop(ys, A, xs)