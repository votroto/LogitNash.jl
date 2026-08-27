include("../src/utils.jl")
include("../src/kernels.jl")
include("../src/tracker.jl")
include("../benchmark/parse_nfg.jl")

using Random
using LinearAlgebra

for fname in readlines()
    dat = read(fname)
    nfg = parse_nfg(dat)

    open("/tmp/path.dat", "w") do io;
        redirect_stdout(io) do;
        explore_manifold( nfg );
        end
    end
    run(`gnuplot ../../test/SETPLOT.222.gp`)
    mv("/tmp/mini.png", "/tmp/$fname.png")
end
