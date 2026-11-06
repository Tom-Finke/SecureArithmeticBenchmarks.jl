# ---------------------------------------------------------------------------
# FHE parameter configurations
# ---------------------------------------------------------------------------

"""
    FHEParams(multiplicative_depth, security_level::OpenFHE.SecurityLevel;
              scaling_mod_size, first_mod_size, label)
    FHEParams(multiplicative_depth, ring_dim::Integer;
              scaling_mod_size, first_mod_size, label)

CKKS parameters for one benchmark context. A context is described by its
multiplicative depth plus *either* a security level or an explicit ring dimension:

    FHEParams(10, OpenFHE.HEStd_128_classic)  # OpenFHE picks the smallest secure ring dim
    FHEParams(10, 1 << 5)                     # explicit ring dim, security `HEStd_NotSet`

With a security level other than `HEStd_NotSet`, `ring_dim` stays `0` and OpenFHE chooses
the ring dimension; the effective value is read back from the generated context.

`scaling_mod_size` and `first_mod_size` are optional and left to OpenFHE when `0`. Together
with the depth they are the levers on rotation key size, which is roughly

    dnum * 2 * (k + ceil(k / dnum)) * ring_dim * 8 bytes

per key, with `k` the number of RNS limbs (≈ `multiplicative_depth + 1`) and `dnum` the
number of hybrid key-switching digits. Measured at `HEStd_128_classic`, depth 10: 23.0 MiB
per key with OpenFHE's defaults, 20.5 MiB with `scaling_mod_size = 40, first_mod_size = 50`,
and 18.3 MiB if the depth also drops to 8.

`dnum` itself is deliberately left at OpenFHE's default and not exposed: setting it to 1
does shrink a key to `4k` tower-polynomials instead of `8k`, but it makes the auxiliary
modulus `P ≈ Q`, and unless the modulus chain is small the resulting `log(Q·P)` no longer
fits the ring dimension the security level allows — at depth 10 with 50/60 OpenFHE then
silently steps up from 32768 to 65536, losing the saving and doubling the cost of every
operation.
"""
struct FHEParams
    label::String
    multiplicative_depth::Int
    security_level::OpenFHE.SecurityLevel  # HEStd_NotSet when ring_dim is given
    ring_dim::Int                          # 0 = let OpenFHE choose from security_level
    scaling_mod_size::Int                  # 0 = OpenFHE default (50)
    first_mod_size::Int                    # 0 = OpenFHE default (60)
end

# `string(::OpenFHE.SecurityLevel)` throws (CxxWrap enums fall into Base's integer
# `string` path), so map the exported values explicitly.
function security_label(level::OpenFHE.SecurityLevel)
    if level == OpenFHE.HEStd_128_classic
        "128c"
    elseif level == OpenFHE.HEStd_192_classic
        "192c"
    elseif level == OpenFHE.HEStd_256_classic
        "256c"
    elseif level == OpenFHE.HEStd_128_quantum
        "128q"
    elseif level == OpenFHE.HEStd_192_quantum
        "192q"
    elseif level == OpenFHE.HEStd_256_quantum
        "256q"
    elseif level == OpenFHE.HEStd_NotSet
        "notset"
    else
        "unknown"
    end
end

# Suffix distinguishing the optional knobs in labels (and therefore in the CSV).
function params_suffix(scaling_mod_size, first_mod_size)
    suffix = ""
    scaling_mod_size > 0 && (suffix *= "_s$(scaling_mod_size)")
    first_mod_size > 0   && (suffix *= "_f$(first_mod_size)")
    suffix
end

# Inverse of `security_label`, for passing a security level to a child process.
function security_level_from_label(label::AbstractString)
    label == "128c"   ? OpenFHE.HEStd_128_classic :
    label == "192c"   ? OpenFHE.HEStd_192_classic :
    label == "256c"   ? OpenFHE.HEStd_256_classic :
    label == "128q"   ? OpenFHE.HEStd_128_quantum :
    label == "192q"   ? OpenFHE.HEStd_192_quantum :
    label == "256q"   ? OpenFHE.HEStd_256_quantum :
    label == "notset" ? OpenFHE.HEStd_NotSet :
    throw(ArgumentError("unknown security level \"$label\""))
end

function FHEParams(multiplicative_depth::Integer, security_level::OpenFHE.SecurityLevel;
                   scaling_mod_size::Integer = 0, first_mod_size::Integer = 0,
                   label = "d$(multiplicative_depth)_$(security_label(security_level))" *
                           params_suffix(scaling_mod_size, first_mod_size))
    FHEParams(label, multiplicative_depth, security_level, 0,
              scaling_mod_size, first_mod_size)
end

function FHEParams(multiplicative_depth::Integer, ring_dim::Integer;
                   scaling_mod_size::Integer = 0, first_mod_size::Integer = 0,
                   label = "d$(multiplicative_depth)_n$(ring_dim)" *
                           params_suffix(scaling_mod_size, first_mod_size))
    FHEParams(label, multiplicative_depth, OpenFHE.HEStd_NotSet, ring_dim,
              scaling_mod_size, first_mod_size)
end

# The swept sizes themselves live in `config.jl`.
const MatrixSize = Union{Int, NTuple{3, Int}}

matrix_dims(d::Int) = (d, d, d)
matrix_dims(t::NTuple{3, Int}) = t

function size_label(sz::MatrixSize)
    n, m, l = matrix_dims(sz)
    sz isa Int ? "$(n)×$(n)" : "$(n)×$(m) * $(m)×$(l)"
end

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# `DEFAULT_FHE_PARAMS` and `DEFAULT_MATRIX_SIZES` are defined in `config.jl`.
struct BenchmarkConfig
    fhe_params::Vector{FHEParams}
    matrix_sizes::Vector{MatrixSize}
    modes::Vector{Symbol}    # :secure_secure, :secure_plain, :plain_secure
    # Run each benchmark once before timing to exclude JIT compilation overhead.
    # See https://docs.julialang.org/en/v1/devdocs/eval/
    warmup::Bool
end

function BenchmarkConfig(;
    fhe_params = DEFAULT_FHE_PARAMS,
    matrix_sizes = DEFAULT_MATRIX_SIZES,
    modes = [:secure_secure, :secure_plain, :plain_secure],
    warmup = true,
)
    BenchmarkConfig(fhe_params, matrix_sizes, modes, warmup)
end

# ---------------------------------------------------------------------------
# Key generation and memory reporting
# ---------------------------------------------------------------------------

# Current resident set size in MiB. `Sys.maxrss()` is a peak rather than a current value,
# so prefer /proc where it exists.
function rss_mib()
    try
        pages = parse(Int, split(read("/proc/self/statm", String))[2])
        round(pages * 4096 / 2^20, digits = 1)
    catch
        round(Sys.maxrss() / 2^20, digits = 1)
    end
end

# ---------------------------------------------------------------------------
# Context setup — raw OpenFHE setup, then wrap with CountingCryptoContext
# ---------------------------------------------------------------------------

function make_context(p::FHEParams)
    parameters = OpenFHE.CCParams{OpenFHE.CryptoContextCKKSRNS}()
    OpenFHE.SetSecretKeyDist(parameters, OpenFHE.UNIFORM_TERNARY)
    OpenFHE.SetMultiplicativeDepth(parameters, p.multiplicative_depth)
    # Explicit: OpenFHE's CKKS default is FLEXIBLEAUTOEXT, which consumes an extra level
    OpenFHE.SetScalingTechnique(parameters, OpenFHE.FLEXIBLEAUTO)
    OpenFHE.SetSecurityLevel(parameters, p.security_level)
    # ring_dim == 0 is OpenFHE's sentinel for "derive from the security level", so only
    # override it when a dimension was requested explicitly.
    if p.ring_dim > 0
        OpenFHE.SetRingDim(parameters, p.ring_dim)
    end
    # Optional knobs; left to OpenFHE's defaults when 0. See the `FHEParams` docstring for
    # how they trade precision and depth against rotation key size.
    # The number of hybrid key-switching digits is left at OpenFHE's default on purpose;
    # see the `FHEParams` docstring.
    p.scaling_mod_size > 0 && OpenFHE.SetScalingModSize(parameters, p.scaling_mod_size)
    p.first_mod_size > 0   && OpenFHE.SetFirstModSize(parameters, p.first_mod_size)

    cc = OpenFHE.GenCryptoContext(parameters)
    OpenFHE.Enable(cc, OpenFHE.PKE)
    OpenFHE.Enable(cc, OpenFHE.KEYSWITCH)
    OpenFHE.Enable(cc, OpenFHE.LEVELEDSHE)
    OpenFHE.Enable(cc, OpenFHE.ADVANCEDSHE)

    # Read the effective values from the raw context before wrapping: SecureArithmetic
    # derives its capacity from GetBatchSize(GetEncodingParams(cc)), not from ring_dim/2.
    ring_dim = Int(OpenFHE.GetRingDimension(cc))
    batch_size = Int(OpenFHE.GetBatchSize(OpenFHE.GetEncodingParams(cc)))
    batch_size = batch_size > 0 ? batch_size : ring_dim ÷ 2

    ccc, counters = make_counting_context(cc)

    context = SecureArithmetic.SecureContext(SecureArithmetic.OpenFHEBackend(ccc))
    public_key, private_key = SecureArithmetic.generate_keys(context)

    (context, public_key, private_key, counters, ring_dim, batch_size)
end

# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

struct BenchmarkResult
    fhe_label::String
    matrix_size::String
    mode::Symbol
    wall_time_s::Float64
    alloc_bytes::Int
    eval_add_count::Int
    eval_add_time_s::Float64
    eval_sub_count::Int
    eval_sub_time_s::Float64
    eval_mult_count::Int
    eval_mult_time_s::Float64
    eval_rotate_count::Int
    eval_rotate_time_s::Float64
    eval_negate_count::Int
    eval_negate_time_s::Float64
    resize_count::Int
    resize_time_s::Float64
    result_level::Int
end

function run_single(context, public_key, private_key, counters::OpCounters,
                    sz::MatrixSize, mode::Symbol)
    n, m, l = matrix_dims(sz)
    m1 = rand(n, m)
    m2 = rand(m, l)

    pm1 = SecureArithmetic.PlainMatrix(collect(m1), context)
    pm2 = SecureArithmetic.PlainMatrix(collect(m2), context)

    sm1 = mode in (:secure_secure, :secure_plain) ? SecureArithmetic.encrypt(pm1, public_key) : nothing
    sm2 = mode in (:secure_secure, :plain_secure)  ? SecureArithmetic.encrypt(pm2, public_key) : nothing

    # Keys are generated per context AND MATRIX SIZE. Generating
    # them here would dominate both runtime and memory: OpenFHE's EvalAtIndexKeyGen builds a
    # complete new key map before replacing the old one, so a regeneration transiently needs
    # twice the (multi-GiB) key memory.
    reset!(counters)
    GC.gc()

    stats = @timed begin
        if mode == :secure_secure
            sm1 * sm2
        elseif mode == :secure_plain
            sm1 * pm2
        elseif mode == :plain_secure
            pm1 * sm2
        end
    end

    c = get_counters(counters)
    result_level = SecureArithmetic.level(stats.value)

    BenchmarkResult(
        "",
        size_label(sz),
        mode,
        stats.time,
        stats.bytes,
        c.eval_add.count,    c.eval_add.time_s,
        c.eval_sub.count,    c.eval_sub.time_s,
        c.eval_mult.count,   c.eval_mult.time_s,
        c.eval_rotate.count, c.eval_rotate.time_s,
        c.eval_negate.count,  c.eval_negate.time_s,
        c.resize.count,       c.resize.time_s,
        result_level,
    )
end

# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

function format_alloc(bytes::Int)
    if bytes >= 1 << 30
        @sprintf("%.1f GiB", bytes / (1 << 30))
    elseif bytes >= 1 << 20
        @sprintf("%.1f MiB", bytes / (1 << 20))
    elseif bytes >= 1 << 10
        @sprintf("%.1f KiB", bytes / (1 << 10))
    else
        @sprintf("%d B", bytes)
    end
end

function print_header()
    @printf("%-10s  %-18s  %-16s  %10s  %10s  %8s %10s  %8s %10s  %8s %10s  %8s %10s  %8s %10s  %8s %10s  %6s\n",
        "FHE", "Size", "Mode", "Wall(s)", "Alloc",
        "#Add", "Add(s)", "#Sub", "Sub(s)",
        "#Mult", "Mult(s)", "#Rot", "Rot(s)",
        "#Neg", "Neg(s)", "#Rsz", "Rsz(s)", "Level")
    println(repeat("─", 221))
end

function print_result(r::BenchmarkResult)
    @printf("%-10s  %-18s  %-16s  %10.3f  %10s  %8d %10.3f  %8d %10.3f  %8d %10.3f  %8d %10.3f  %8d %10.3f  %8d %10.3f  %6d\n",
        r.fhe_label, r.matrix_size, r.mode, r.wall_time_s, format_alloc(r.alloc_bytes),
        r.eval_add_count, r.eval_add_time_s,
        r.eval_sub_count, r.eval_sub_time_s,
        r.eval_mult_count, r.eval_mult_time_s,
        r.eval_rotate_count, r.eval_rotate_time_s,
        r.eval_negate_count, r.eval_negate_time_s,
        r.resize_count, r.resize_time_s,
        r.result_level)
end

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

const CSV_HEADER = join([
    "fhe_params", "matrix_size", "mode", "wall_time_s", "alloc_bytes",
    "eval_add_count", "eval_add_time_s",
    "eval_sub_count", "eval_sub_time_s",
    "eval_mult_count", "eval_mult_time_s",
    "eval_rotate_count", "eval_rotate_time_s",
    "eval_negate_count", "eval_negate_time_s",
    "resize_count", "resize_time_s",
    "result_level",
], ",")

function write_csv(results::Vector{BenchmarkResult}, path::String; quiet::Bool = false)
    open(path, "w") do io
        println(io, CSV_HEADER)
        for r in results
            println(io, join([
                r.fhe_label, r.matrix_size, r.mode, r.wall_time_s, r.alloc_bytes,
                r.eval_add_count, r.eval_add_time_s,
                r.eval_sub_count, r.eval_sub_time_s,
                r.eval_mult_count, r.eval_mult_time_s,
                r.eval_rotate_count, r.eval_rotate_time_s,
                r.eval_negate_count, r.eval_negate_time_s,
                r.resize_count, r.resize_time_s,
                r.result_level,
            ], ","))
        end
    end
    quiet || println("\nResults written to $path")
end

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

function run_benchmarks(config::BenchmarkConfig = BenchmarkConfig())
    SecureArithmetic.disable_multithreading()
    results = BenchmarkResult[]

    for p in config.fhe_params
        println("\n━━━ Setting up FHE context: $(p.label) " *
                "(depth=$(p.multiplicative_depth), security=$(security_label(p.security_level))" *
                (p.ring_dim > 0 ? ", ring_dim=$(p.ring_dim)" : "") * ") ━━━")
        context, public_key, private_key, counters, ring_dim, batch_size = make_context(p)
        init_multiplication!(context, private_key)
        println("  ring dimension $(ring_dim), batch size $(batch_size), RSS $(rss_mib()) MiB")
        
        for sz in config.matrix_sizes
            n, m, l = matrix_dims(sz)
            SecureArithmetic.init_matrix_multiplication!(context, private_key, (n,m), (m,l))
            n_elements = max(n * m, m * l)
            ct_label = n_elements <= batch_size ? "single-CT" : "multi-CT"
            println("  Matrix $(size_label(sz)) ($ct_label, $(n_elements) slots / batch " *
                    "$(batch_size), RSS $(rss_mib()) MiB)")

            for mode in config.modes
                if config.warmup
                    run_single(context, public_key, private_key, counters, sz, mode)
                end

                r = run_single(context, public_key, private_key, counters, sz, mode)
                r = BenchmarkResult(
                    p.label, r.matrix_size, r.mode, r.wall_time_s, r.alloc_bytes,
                    r.eval_add_count, r.eval_add_time_s,
                    r.eval_sub_count, r.eval_sub_time_s,
                    r.eval_mult_count, r.eval_mult_time_s,
                    r.eval_rotate_count, r.eval_rotate_time_s,
                    r.eval_negate_count, r.eval_negate_time_s,
                    r.resize_count, r.resize_time_s,
                    r.result_level,
                )
                push!(results, r)
                print_result(r)
            end
        end

        SecureArithmetic.release_context_memory()
        GC.gc()
    end

    results
end
