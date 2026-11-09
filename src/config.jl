# ===========================================================================
#  WHAT THE BENCHMARK SWEEPS — edit this file to change the sweep
# ===========================================================================
#
# `run.jl` benchmarks every combination of the FHE parameters and matrix sizes below, in
# each of the modes `:secure_secure`, `:secure_plain` and `:plain_secure`. Each combination
# runs in its own memory-capped process (see `sweep.jl`).

# ---------------------------------------------------------------------------
# FHE parameters
# ---------------------------------------------------------------------------
#
# A configuration is a multiplicative depth plus *either* a security level or an explicit
# ring dimension:
#
#   FHEParams(10, OpenFHE.HEStd_128_classic)   # OpenFHE picks the smallest secure ring dim
#   FHEParams(10, 1 << 5)                      # explicit ring dim, security HEStd_NotSet
#
# Optional keywords `scaling_mod_size` (default 50) and `first_mod_size` (default 60) trade
# precision per level against rotation key size, which dominates the memory use of a sweep:
#
#   FHEParams(8, OpenFHE.HEStd_128_classic; scaling_mod_size = 40, first_mod_size = 50)
#
# At HEStd_128_classic and depth 10 (ring dimension 32768) a 28×28 configuration needs 165
# rotation keys of ~23 MiB each, i.e. ~3.7 GiB; the keywords above bring that to ~2.9 GiB.
# See the `FHEParams` docstring in `benchmark_matmul.jl` for the full size formula.
const DEFAULT_FHE_PARAMS = [
    FHEParams(15, 2 << 10),
    FHEParams(10, OpenFHE.HEStd_128_classic),
]

# ---------------------------------------------------------------------------
# Matrix sizes
# ---------------------------------------------------------------------------
#
# `d` means the square product `d×d * d×d`; `(n, m, l)` means `n×m * m×l`, where the first
# operand must be square (`n == m`) and `l` must divide `m`. Rotation key count grows as
# `3d - 2` shifts, so larger `d` costs proportionally more memory and key generation time.
const DEFAULT_MATRIX_SIZES = MatrixSize[
    4, (4, 4, 1),
    8, (8, 8, 1),
    16, (16, 16, 1),
    # 28, (28, 28, 1),
]
