# LogitNash.jl

![Coverage](https://gist.githubusercontent.com/votroto/7cc2561e5897bc8c43270ce659b45199/raw/LogitNashCoverage.svg)

> [!WARNING]
> Before going out of pre-release, the API must be finalized. If you have suggestions, please get in touch!

Approximates a specific mixed Nash equilibrium of a multiplayer general-sum game up to a desired precision by tracing the principal logit equilibrium branch

$$\pi_p \propto \left(e^{\lambda U_p^i(\pi_{-p})}\right)_i$$

The algorithm stops when either:
 - ($\epsilon$-NE condition) there is no unilateral deviation from $\pi$ more profitable than `stop_eps`,
 - (logit condition) a solution is found for the precision parameter $\lambda$ greater or equal to `stop_lambda`,
 - or the maximum number of iterations of `stop_iters` is reached.

## Speed and Stability

*LogitNash.jl* currently achieves a speedup of **82×** compared to *gambit-logit*! Check our our [Benchmarks](https://github.com/votroto/LogitNash.jl/wiki/Benchmarks) wiki page for details.

 - Measured the time to reach $\lambda = 10^6$ over 100 samples from `RandomGame` GAMUT distribution with 5 actions and 6 players using a single thread of Xeon Gold 6146.

 Not convinced? Benchmark it yourself with the prepared `/benchmark` scripts for Bash or Slurm.

## Install

Install the package directly from GitHub by running:
```julia
using Pkg; Pkg.add(url="https://github.com/votroto/LogitNash.jl")
```

## Usage

The following example solves a three-player prisoners dilemma
```julia
using LogitNash

p1 = Float64[3 1; 5 2;;; 1 0; 2 1]
p2 = Float64[3 5; 1 2;;; 1 2; 0 1]
p3 = Float64[3 1; 1 0;;; 5 2; 2 1]
three_prisoners = (p1, p2, p3)

pi, status = nash(three_prisoners; stop_iters=1000, stop_t=1e6, stop_eps=1e-6)

for p in eachindex(pi)
    println("pi_$p = ", round.(pi[p]; digits=5))
end
```
which should result in the strategy profile
```julia
pi_1 = [0.0, 1.0]
pi_2 = [0.0, 1.0]
pi_3 = [0.0, 1.0]
```

## Acknowledgements

Continued development and solver comparison benchmarks are possible thanks to:
- The support by the Czech Science Foundation grant--no. 24-12046S.
- RCI providing access to the computational infrastructure of the OP VVV funded project CZ.02.1.01/0.0/0.0/16_019/0000765.

Many thanks go to `BifurcationKit.jl`, `HomotopyContinuation.jl`, `bertini`, `gambit-logit` for their source code and manuals; to ChatGPT for deriving the jacobians; to Gemini for source-porting a lot of testing utilities, and GitHub Copilot for refactoring; to Mosek, Gurobi and PATH for their various available licences for testing, and to AIC for their inexplicable continued support.

Based on the papers:

> Turocy, T. L. (2005). A dynamic homotopy interpretation of the logistic quantal response equilibrium correspondence. *Games and Economic Behavior, 51*(2), 243–263. https://doi.org/10.1016/j.geb.2004.04.003

> Timme, S. (2021). Mixed precision path tracking for polynomial homotopy continuation. Advances in Computational Mathematics, 47, 75. https://doi.org/10.1007/s10444-021-09899-y
