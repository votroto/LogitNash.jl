include("../src/spaces.jl")
include("../src/turocy.jl")
include("../src/path.jl")
include("../benchmark/parse_nfg.jl")

using Printf
using Random
using LinearAlgebra


function print_strats222(ss)
    println(join([@sprintf("%.5f", first(ss[p])) for p in eachindex(ss)], " "))
end

function print_strats(ss)
    for p in eachindex(ss)
        println(join([@sprintf("%.5f", s) for s in ss[p]], " "))
    end
end


for fname in readlines()
    dat = read(fname)
    nfg = parse_nfg(dat)

    println(fname)
    ne, st = nash( nfg; stop_eps=1e-9, stop_t=1e11);
    print_strats(ne)
end

#=

open("/tmp/sol.dat", "w") do io; redirect_stdout(io) do; print_strats222(nash( nfg; stop_eps=1e-9, stop_t=1e11)[1]); end end

=#

function writepathsol(nfg)
    open("/tmp/path.dat", "w") do io; redirect_stdout(io) do; explore_manifold( nfg ); end end
    open("/tmp/sol.dat", "w") do io; redirect_stdout(io) do; print_strats222(nash( nfg; stop_eps=1e-9, stop_t=1e11)[1]); end end
end
