"""
    run.jl — CLI entry point for SecureArithmetic matrix multiplication benchmarks.

Run from the project root:

    OMP_NUM_THREADS=1 julia --project run.jl --secure-arithmetic-path /path/to/SecureArithmetic.jl

To write results to CSV:

    OMP_NUM_THREADS=1 julia --project run.jl --secure-arithmetic-path /path/to/SecureArithmetic.jl --csv results.csv

Each `(FHE params, matrix size)` combination runs in its own process inside a transient
systemd scope with a hard memory limit, so a configuration whose rotation keys do not fit is
killed by the kernel and reported as "over limit" while the rest of the sweep continues. The
limit defaults to 60% of total system memory and can be set explicitly:

    OMP_NUM_THREADS=1 julia --project run.jl --max-memory 6G
    OMP_NUM_THREADS=1 julia --project run.jl --max-memory 40%
    OMP_NUM_THREADS=1 julia --project run.jl --max-memory 0     # no limit

A run that raises an exception instead - for instance a multiplicative depth too small for
the matrix size - does not abort anything: it is printed as `✗ errored: <reason>`, written to
the CSV as a row with `status = error`, and the remaining modes, sizes and parameter sets
still run. The sweep's exit code is 1 if any configuration errored, was killed or failed.

Which FHE parameters and matrix sizes are swept is defined in `src/config.jl`.

Note: `--secure-arithmetic-path` calls `Pkg.develop()`, which on Julia 1.11+ writes a
`[sources]` section into `Project.toml`. This is local state and should not be committed.

A Julia process with OpenFHE loaded occupies ~600 MiB before generating a single key, and
the cgroup reclaims file-backed pages rather than killing the process, so limits below
roughly 1 GiB do not constrain this workload in a useful way. Note also that the default is
a fraction of *total* memory: on a machine where other applications already hold several
GiB, pass an explicit limit that fits what is actually free.

`--child --label … --depth … --security … --size … --csv fragment.csv` runs a single
configuration and is used internally by the sweep, which passes the full configuration so
that any `BenchmarkConfig` works, not only the default one. The child never calls
`Pkg.develop`, so children cannot race on `Project.toml`.
"""

function flag_value(name::String)
    for (i, arg) in enumerate(ARGS)
        arg == name && i < length(ARGS) && return ARGS[i + 1]
    end
    nothing
end

const IS_CHILD = "--child" in ARGS
const CSV_PATH = flag_value("--csv")

if !IS_CHILD
    import Pkg
    sa_path = flag_value("--secure-arithmetic-path")
    if sa_path !== nothing
        # Updates Manifest.toml in-place; fast no-op when the path hasn't changed
        Pkg.develop(path = sa_path)
    end
end

using SecureArithmeticBenchmarks

if IS_CHILD
    dims = Tuple(parse.(Int, split(flag_value("--size"), ",")))
    params = FHEParams(flag_value("--label"),
                       parse(Int, flag_value("--depth")),
                       SecureArithmeticBenchmarks.security_level_from_label(
                           flag_value("--security")),
                       parse(Int, flag_value("--ring-dim")),
                       parse(Int, flag_value("--scaling-mod")),
                       parse(Int, flag_value("--first-mod")))

    config = BenchmarkConfig(fhe_params = [params],
                             # `--square` preserves whether the size was given as a single
                             # dimension, which decides how the result is labelled.
                             matrix_sizes = MatrixSize[flag_value("--square") == "true" ?
                                                       dims[1] : dims],
                             modes = Symbol.(split(flag_value("--modes"), ",")),
                             warmup = flag_value("--warmup") == "true")

    # The parent prints the table header and merges the fragments.
    results = run_benchmarks(config)
    CSV_PATH === nothing || SecureArithmeticBenchmarks.write_csv(results, CSV_PATH,
                                                                 quiet = true)
    # The fragment is written either way; the exit code tells the parent whether to report
    # this configuration as partial.
    exit(any(r -> r.status === :error, results) ?
         SecureArithmeticBenchmarks.PARTIAL_EXIT_CODE : 0)
end

println("SecureArithmetic Matrix Multiplication Benchmark")
println("================================================")
println("Tip: run with OMP_NUM_THREADS=1 for accurate per-op timing.\n")

max_memory = flag_value("--max-memory")
skipped = run_sweep(BenchmarkConfig(); csv = CSV_PATH, max_memory = max_memory)

exit(isempty(skipped) ? 0 : 1)
