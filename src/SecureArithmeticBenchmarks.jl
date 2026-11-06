module SecureArithmeticBenchmarks

using SecureArithmetic
using OpenFHE
using Printf

export CountingCryptoContext, OpCounters
export reset!, get_counters, make_counting_context
export FHEParams, MatrixSize, BenchmarkConfig, run_benchmarks
export run_sweep, default_memory_limit, parse_memory_limit

include("counting_context.jl")
include("benchmark_matmul.jl")
# What the sweep covers — edit `config.jl` to change the FHE parameters and matrix sizes.
# It comes after `benchmark_matmul.jl`, which defines the `FHEParams` and `MatrixSize` types
# it uses.
include("config.jl")
include("sweep.jl")

end
