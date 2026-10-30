"""
    run.jl — CLI entry point for SecureArithmetic matrix multiplication benchmarks.

Run from the project root:

    OMP_NUM_THREADS=1 julia --project run.jl --secure-arithmetic-path /path/to/SecureArithmetic.jl

To write results to CSV:

    OMP_NUM_THREADS=1 julia --project run.jl --secure-arithmetic-path /path/to/SecureArithmetic.jl --csv results.csv

Note: `--secure-arithmetic-path` calls `Pkg.develop()`, which on Julia 1.11+ writes a
`[sources]` section into `Project.toml`. This is local state and should not be committed.
"""

import Pkg

sa_path = nothing
csv_path = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--secure-arithmetic-path" && i < length(ARGS)
        global sa_path = ARGS[i + 1]
    elseif arg == "--csv" && i < length(ARGS)
        global csv_path = ARGS[i + 1]
    end
end

if sa_path !== nothing
    # Updates Manifest.toml in-place; fast no-op when the path hasn't changed
    Pkg.develop(path=sa_path)
end

using SecureArithmeticBenchmarks

println("SecureArithmetic Matrix Multiplication Benchmark")
println("================================================")
println("Tip: run with OMP_NUM_THREADS=1 for accurate per-op timing.\n")

SecureArithmeticBenchmarks.print_header()
results = run_benchmarks()

if csv_path !== nothing
    SecureArithmeticBenchmarks.write_csv(results, csv_path)
end

println("\nDone. $(length(results)) benchmarks completed.")
