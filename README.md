# LogitNash.jl

![Coverage](https://gist.githubusercontent.com/votroto/7cc2561e5897bc8c43270ce659b45199/raw/LogitNashCoverage.svg)

> [!WARNING]
> Before going out of pre-release, the API must be finalized. If you have suggestions, please get in touch!

Approximates a specific mixed Nash equilibrium of a multiplayer general-sum game up to a desired precision by tracing the [principal logit equilibrium branch](https://github.com/votroto/LogitNash.jl/wiki) parametrized by $\lambda$

$$\pi_p \propto \left(e^{\lambda U_p^i(\pi_{-p})}\right)_i.$$

The algorithm stops when either:
 - ($\epsilon$-NE condition) there is no unilateral deviation from $\pi$ more profitable than `stop_eps`,
 - (logit condition) a solution is found for the precision parameter $\lambda$ greater or equal to `stop_lambda`,
 - or the maximum number of iterations of `stop_iters` is reached.

## Speed and Stability

*LogitNash.jl* can be orders of magnitude faster than *gambit-logit*! See our [Benchmarks](https://github.com/votroto/LogitNash.jl/wiki/Benchmarks) for details and the full *GAMUT*.

Not convinced? Benchmark it yourself with the prepared `/benchmark` scripts for Bash or Slurm.

## Install

Install the package directly from GitHub by running
```julia
using Pkg; Pkg.add(url="https://github.com/votroto/LogitNash.jl")
```

## Usage

The following example solves a three-player prisoners dilemma
```julia
using LogitNash

u1 = Float64[3 1; 5 2;;; 1 0; 2 1]
u2 = Float64[3 5; 1 2;;; 1 2; 0 1]
u3 = Float64[3 1; 1 0;;; 5 2; 2 1]
three_prisoners = (u1, u2, u3)

pi, status = solve(three_prisoners; stop_iters=1000, stop_lambda=1e6, stop_eps=1e-6)

for p in eachindex(pi)
    println("pi_$p = ", round.(pi[p]; digits=5))
end
```
and shows the strategy profile
```julia
pi_1 = [0.0, 1.0]
pi_2 = [0.0, 1.0]
pi_3 = [0.0, 1.0]
```

## Acknowledgements

Continued development and solver comparison benchmarks are possible thanks to:
- the support by the Czech Science Foundation grant--no. 24-12046S,
- the computational infrastructure of the OP VVV funded project CZ.02.1.01/0.0/0.0/16_019/0000765 access provided by RCI.

Many thanks go to `BifurcationKit.jl`, `HomotopyContinuation.jl`, *Bertini*, and *Gambit* for their source code and manuals; to ChatGPT for deriving the jacobians; to Gemini for source-porting a lot of testing utilities, and GitHub Copilot for refactoring; to Mosek, Gurobi and PATH for their various available licences for testing, and to AIC for their inexplicable continued support.

Based on the papers:

- Turocy, T. L. (2005). A dynamic homotopy interpretation of the logistic quantal response equilibrium correspondence. *Games and Economic Behavior, 51*(2), 243–263. https://doi.org/10.1016/j.geb.2004.04.003
- Timme, S. (2021). Mixed precision path tracking for polynomial homotopy continuation. Advances in Computational Mathematics, 47, 75. https://doi.org/10.1007/s10444-021-09899-y