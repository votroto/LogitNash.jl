
## Notes
- `generators.conf` entries are lines containing the *dataset-name* followed by a command to generate one game.
- Gambit can stop once it reaches a desired regret, or try to reach a given lambda, not both.
- The Kernels are specialized per the number of players. The first time an N-player game is solved will incur a compilation time penalty.
- There is no specialization for zero-sum games. A reasonable linear program will always be faster.
- The implementation of the algorithm should be tested in isolation, that is: without scaling, pruning, or perturbation. Even once such features are exposed, they should be disabled by default. Unfortunately, since we don't have any *naturally* poorly scaled dataset, and Gambit scales by default, all the games from GAMUT are normalized to keep the comparison fair.
- The project is a work-in-progress.

## Very Rough Benchmark
Time in seconds to reach regret 1e-6 on `randn` (`Float64`) games, *AMD Ryzen 7 PRO 5850U*, 2026-08-30 (Limited to *L3* size, 5 samples per game):

|       |   2 act. |   3 act. |   4 act. |   5 act. |   6 act. |   7 act. |   8 act. |   9 act. |  10 act. |  15 act. |  20 act. |  30 act. |  40 act. |  50 act. |
|------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| **2** |   0.0001 |   0.0002 |   0.0002 |   0.0002 |   0.0004 |   0.0005 |   0.0007 |   0.0009 |   0.0011 |   0.0030 |   0.0043 |   0.0089 |   0.0231 |   0.0271 |
| **3** |   0.0002 |   0.0004 |   0.0006 |   0.0009 |   0.0015 |   0.0024 |   0.0025 |   0.0034 |   0.0039 |   0.0146 |   0.0264 |   0.0875 |   0.1340 |   0.4042 |
| **4** |   0.0003 |   0.0007 |   0.0014 |   0.0027 |   0.0048 |   0.0087 |   0.0105 |   0.0148 |   0.0233 |   0.1302 |   0.3311 |          |          |          |
| **5** |   0.0007 |   0.0021 |   0.0060 |   0.0147 |   0.0379 |   0.0906 |   0.1175 |   0.1965 |   0.3950 |
| **6** |   0.0013 |   0.0071 |   0.0271 |   0.0992 |   0.2800 |   0.6828 |          |          |          |
| **7** |   0.0030 |   0.0259 |   0.1508 |   0.7766 |          |          |          |          |          |
| **8** |   0.0058 |   0.0953 |   0.7423 |          |          |          |          |          |          |
| **9** |   0.0167 |   0.3976 |          |          |          |          |          |          |          |

## Dependencies

- To compare against Gambit, [download it here and compile it](https://github.com/gambitproject/gambit) or get it from a package manager.
- Dataset generator [GAMUT must be downloaded here](http://gamut.stanford.edu/)
> Nudelman, Eugene & Wortman, Jennifer & Shoham, Yoav & Leyton-Brown, Kevin. (2004). Run the GAMUT: A Comprehensive Approach to Evaluating Game-Theoretic Algorithms.. Proceedings of the Third International Joint Conference on Autonomous Agents and Multiagent Systems. 2. 880-887. 10.1109/AAMAS.2004.238.

