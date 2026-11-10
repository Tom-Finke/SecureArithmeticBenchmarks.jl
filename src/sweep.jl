# ---------------------------------------------------------------------------
# Memory-capped sweep: one child process per (FHE params, matrix size)
# ---------------------------------------------------------------------------
#
# Rotation keys dominate the memory use of a sweep — at ring dimension 32768 and depth 10 a
# single 28×28 configuration needs 165 keys of ~23 MiB each, i.e. ~3.7 GiB — and OpenFHE
# keeps them in static maps that only `release_context_memory()` frees, which invalidates
# the context along with them. Running each configuration in its own process therefore both
# bounds the memory and makes a configuration that does not fit a recorded result rather
# than a dead machine: the child runs inside a transient systemd scope with a hard
# `MemoryMax`, so the kernel kills that child alone and the sweep continues.

const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
const RUN_SCRIPT = joinpath(PROJECT_DIR, "run.jl")

# A child killed by the cgroup OOM killer either reports termsignal 9 or exits with 137.
const SIGKILL_EXIT_CODE = 137

# A child that ran to completion but had at least one errored run exits with this code. It
# still wrote its CSV fragment, so its rows - including the errored ones - are kept and the
# configuration is reported as partial rather than failed.
const PARTIAL_EXIT_CODE = 3

"""
    mem_total_bytes() -> Int

Total system memory in bytes, read from `/proc/meminfo`.
"""
function mem_total_bytes()
    for line in eachline("/proc/meminfo")
        if startswith(line, "MemTotal:")
            return parse(Int, split(line)[2]) * 1024
        end
    end
    error("could not read MemTotal from /proc/meminfo")
end

"""
    default_memory_limit() -> Int

Default per-configuration memory limit: 60% of total system memory, leaving room for the
rest of the system.
"""
default_memory_limit() = (mem_total_bytes() * 3) ÷ 5

"""
    parse_memory_limit(spec) -> Int

Parse a memory limit into bytes. Accepts `"6G"`, `"6GiB"`, `"512M"`, `"60%"` (of total
system memory) and plain byte counts. Suffixes are binary (`1G == 1024³`).
"""
function parse_memory_limit(spec::AbstractString)
    s = uppercase(strip(spec))
    if endswith(s, "%")
        percent = parse(Float64, chop(s))
        0 < percent <= 100 || throw(ArgumentError("memory limit percentage out of range: $spec"))
        return round(Int, mem_total_bytes() * percent / 100)
    end
    m = match(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)(?:I?B)?$", s)
    m === nothing && throw(ArgumentError("cannot parse memory limit \"$spec\""))
    multiplier = Dict("" => 1, "K" => 1024, "M" => 1024^2,
                      "G" => 1024^3, "T" => 1024^4)[m.captures[2]]
    round(Int, parse(Float64, m.captures[1]) * multiplier)
end

parse_memory_limit(bytes::Integer) = Int(bytes)

const SYSTEMD_RUN_AVAILABLE = Ref{Union{Nothing, Bool}}(nothing)

"""
    systemd_run_available() -> Bool

Whether transient scopes with a memory limit can be started, i.e. `systemd-run` exists and
a systemd user manager is reachable. The result is cached.
"""
function systemd_run_available()
    if SYSTEMD_RUN_AVAILABLE[] === nothing
        SYSTEMD_RUN_AVAILABLE[] =
            Sys.which("systemd-run") !== nothing &&
            success(pipeline(`systemctl --user show -p Version`;
                             stdout = devnull, stderr = devnull))
    end
    SYSTEMD_RUN_AVAILABLE[]
end

# The child is given the configuration itself rather than an index into the defaults, so
# `run_sweep` works for any `BenchmarkConfig`, not only `BenchmarkConfig()`.
spec_size(sz::MatrixSize) = join(matrix_dims(sz), ",")
spec_modes(modes) = join(modes, ",")

"""
    child_command(p, sz, modes, warmup, fragment, limit_bytes) -> Cmd

Command running one configuration in a separate process. With `limit_bytes > 0` the process
is placed in a transient systemd scope whose `MemoryMax` the kernel enforces.
"""
function child_command(p::FHEParams, sz::MatrixSize, modes, warmup::Bool,
                       fragment::AbstractString, limit_bytes::Integer)
    julia = joinpath(Sys.BINDIR, "julia")
    inner = `$julia --project=$PROJECT_DIR $RUN_SCRIPT --child
             --label $(p.label) --depth $(p.multiplicative_depth)
             --security $(security_label(p.security_level)) --ring-dim $(p.ring_dim)
             --scaling-mod $(p.scaling_mod_size) --first-mod $(p.first_mod_size)
             --size $(spec_size(sz)) --square $(sz isa Int)
             --modes $(spec_modes(modes)) --warmup $(warmup)
             --csv $fragment`
    limit_bytes > 0 || return inner
    # --scope moves systemd-run itself into the new cgroup and then execs, so the limit
    # applies to the Julia process and the environment is inherited. --collect removes the
    # unit afterwards, including when the child was killed.
    `systemd-run --user --scope --quiet --collect
      -p MemoryMax=$limit_bytes -p MemorySwapMax=0 -- $inner`
end

function merge_fragments(fragments::Vector{String}, path::AbstractString)
    open(path, "w") do io
        println(io, CSV_HEADER)
        for fragment in fragments
            for (i, line) in enumerate(eachline(fragment))
                i == 1 && continue   # the fragment's own header
                println(io, line)
            end
        end
    end
    println("\nResults written to $path")
end

"""
    run_sweep(config = BenchmarkConfig(); csv = nothing, max_memory = nothing)

Run every `(FHE params, matrix size)` combination of `config` in its own memory-capped
process and return the configurations that did not complete, as
`(fhe_label, matrix_size, reason)` tuples.

`max_memory` accepts anything [`parse_memory_limit`](@ref) understands and defaults to
[`default_memory_limit`](@ref); pass `0` to run without a limit. Each child prints its
result rows as it goes. With `csv` given, the children's CSV fragments are merged into that
path; a configuration that was killed contributes no rows.

A run that throws is handled inside the child (see [`run_benchmarks`](@ref)): it becomes a
row with `status = error`, its siblings still run, and the child exits with
`PARTIAL_EXIT_CODE`. Such a configuration keeps its rows and is listed here as
`"some runs errored"`, so the returned vector is non-empty and the process exit code of
`run.jl` signals that the sweep was not clean.
"""
function run_sweep(config::BenchmarkConfig = BenchmarkConfig();
                   csv::Union{Nothing, AbstractString} = nothing,
                   max_memory = nothing)
    limit = max_memory === nothing ? default_memory_limit() : parse_memory_limit(max_memory)

    if limit > 0 && !systemd_run_available()
        @warn "systemd-run or the systemd user manager is unavailable - running " *
              "configurations without a memory limit"
        limit = 0
    end
    if limit > 0
        println("Memory limit per configuration: $(format_alloc(limit)) " *
                "(cgroup MemoryMax, enforced by the kernel)")
    else
        println("Memory limit per configuration: none")
    end

    print_header()

    fragment_dir = mktempdir()
    fragments = String[]
    failed = Tuple{String, String, String}[]
    partial = Tuple{String, String, String}[]

    for (i, p) in enumerate(config.fhe_params), (j, sz) in enumerate(config.matrix_sizes)
        fragment = joinpath(fragment_dir, "fragment_$(i)_$(j).csv")
        process = run(ignorestatus(child_command(p, sz, config.modes, config.warmup,
                                                 fragment, limit)))

        if process.exitcode in (0, PARTIAL_EXIT_CODE) && isfile(fragment)
            push!(fragments, fragment)
            if process.exitcode == PARTIAL_EXIT_CODE
                push!(partial, (p.label, size_label(sz), "some runs errored"))
            end
        else
            killed = process.termsignal == 9 || process.exitcode == SIGKILL_EXIT_CODE
            reason = killed ? "over limit ($(format_alloc(limit)))" :
                              "failed (exit code $(process.exitcode))"
            push!(failed, (p.label, size_label(sz), reason))
            println("  ✗ $(p.label)  $(size_label(sz)): $reason")
        end
    end

    csv === nothing || merge_fragments(fragments, csv)

    n_clean = length(fragments) - length(partial)
    if isempty(failed) && isempty(partial)
        println("\nAll $(n_clean) configuration(s) completed.")
    else
        println("\n$(n_clean) configuration(s) completed cleanly, " *
                "$(length(partial)) with errored runs, " *
                "$(length(failed)) did not complete:")
        for (label, size, reason) in Iterators.flatten((partial, failed))
            println("  $(rpad(label, 16)) $(rpad(size, 18)) $reason")
        end
    end

    vcat(partial, failed)
end
