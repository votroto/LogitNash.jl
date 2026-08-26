
## Notes
- The Jacobian Kernels are specialized per the number of players. The first time an N-player game is solved will incur a compilation time penalty.
- There is no specialization for zero-sum games. A reasonable linear program will always be faster. (In fact the only exposed implementation is for dense tensors of 64-bit floats.)
- The project is a work-in-progress.