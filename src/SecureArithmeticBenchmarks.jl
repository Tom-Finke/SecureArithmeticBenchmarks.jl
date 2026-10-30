module SecureArithmeticBenchmarks

using SecureArithmetic
using OpenFHE
using Printf

export CountingCryptoContext, OpCounters
export reset!, get_counters, make_counting_context
export FHEParams, MatrixSize, BenchmarkConfig, run_benchmarks

include("counting_context.jl")
include("benchmark_matmul.jl")

end
