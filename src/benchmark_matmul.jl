# ---------------------------------------------------------------------------
# FHE parameter configurations
# ---------------------------------------------------------------------------

struct FHEParams
    label::String
    ring_dim::Int
    scaling_mod_size::Int
    first_mod_size::Int
    levels_after_bootstrap::Int
    level_budget::Vector{Int}
end

# FHEParams(label, ring_dim, scaling_mod_size, first_mod_size,
#           levels_after_bootstrap, level_budget)
#   ring_dim            – polynomial ring dimension (power of 2); batch_size = ring_dim / 2
#   scaling_mod_size    – bit-size of the CKKS scaling modulus (controls precision per level)
#   first_mod_size      – bit-size of the first (largest) modulus in the RNS chain
#   levels_after_bootstrap – multiplicative levels available after bootstrapping
#   level_budget        – [CtS, StC] level budget for the two bootstrapping phases
# These default values are taken from the examples in SecureArithmetic
const DEFAULT_FHE_PARAMS = [
    FHEParams("ring2^5", 1 << 5, 59, 60, 10, [4, 4]),
]

const MatrixSize = Union{Int, NTuple{3, Int}}

matrix_dims(d::Int) = (d, d, d)
matrix_dims(t::NTuple{3, Int}) = t

const DEFAULT_MATRIX_SIZES = MatrixSize[
    3, 4, 8, 9, 32,
    (3,3,1),
    (4,4,1),
    (8,8,1),
    (9,9,1),
    (32,32,1),
]

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

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
# Context setup — raw OpenFHE setup, then wrap with CountingCryptoContext
# ---------------------------------------------------------------------------

function make_context(p::FHEParams)
    depth = p.levels_after_bootstrap +
            OpenFHE.GetBootstrapDepth(p.level_budget, OpenFHE.UNIFORM_TERNARY)

    parameters = OpenFHE.CCParams{OpenFHE.CryptoContextCKKSRNS}()
    OpenFHE.SetSecretKeyDist(parameters, OpenFHE.UNIFORM_TERNARY)
    OpenFHE.SetSecurityLevel(parameters, OpenFHE.HEStd_NotSet)
    OpenFHE.SetRingDim(parameters, p.ring_dim)
    OpenFHE.SetScalingModSize(parameters, p.scaling_mod_size)
    OpenFHE.SetScalingTechnique(parameters, OpenFHE.FLEXIBLEAUTO)
    OpenFHE.SetFirstModSize(parameters, p.first_mod_size)
    OpenFHE.SetMultiplicativeDepth(parameters, depth)

    cc = OpenFHE.GenCryptoContext(parameters)
    OpenFHE.Enable(cc, OpenFHE.PKE)
    OpenFHE.Enable(cc, OpenFHE.KEYSWITCH)
    OpenFHE.Enable(cc, OpenFHE.LEVELEDSHE)
    OpenFHE.Enable(cc, OpenFHE.ADVANCEDSHE)
    OpenFHE.Enable(cc, OpenFHE.FHE)

    OpenFHE.EvalBootstrapSetup(cc; level_budget = p.level_budget)

    ccc, counters = make_counting_context(cc)

    context = SecureArithmetic.SecureContext(SecureArithmetic.OpenFHEBackend(ccc))
    public_key, private_key = SecureArithmetic.generate_keys(context)

    (context, public_key, private_key, counters)
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
    multiplicative_depth::Int
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

    max_elems = max(n * m, m * l)
    SecureArithmetic.init_rotation!(context, private_key, max_elems, 1:max_elems...)
    SecureArithmetic.init_multiplication!(context, private_key)

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
    depth = SecureArithmetic.level(stats.value)

    size_label = sz isa Int ? "$(n)×$(n)" : "$(n)×$(m) * $(m)×$(l)"

    BenchmarkResult(
        "",
        size_label,
        mode,
        stats.time,
        stats.bytes,
        c.eval_add.count,    c.eval_add.time_s,
        c.eval_sub.count,    c.eval_sub.time_s,
        c.eval_mult.count,   c.eval_mult.time_s,
        c.eval_rotate.count, c.eval_rotate.time_s,
        c.eval_negate.count,  c.eval_negate.time_s,
        c.resize.count,       c.resize.time_s,
        depth,
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
        "#Neg", "Neg(s)", "#Rsz", "Rsz(s)", "Depth")
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
        r.multiplicative_depth)
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
    "multiplicative_depth",
], ",")

function write_csv(results::Vector{BenchmarkResult}, path::String)
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
                r.multiplicative_depth,
            ], ","))
        end
    end
    println("\nResults written to $path")
end

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

function run_benchmarks(config::BenchmarkConfig = BenchmarkConfig())
    SecureArithmetic.disable_multithreading()
    results = BenchmarkResult[]

    for p in config.fhe_params
        println("\n━━━ Setting up FHE context: $(p.label) (ring_dim=$(p.ring_dim)) ━━━")
        context, public_key, private_key, counters = make_context(p)
        batch_size = p.ring_dim ÷ 2

        for sz in config.matrix_sizes
            n, m, l = matrix_dims(sz)
            n_elements = max(n * m, m * l)
            ct_label = n_elements <= batch_size ? "single-CT" : "multi-CT"
            dims_str = sz isa Int ? "$(n)×$(n)" : "$(n)×$(m) * $(m)×$(l)"
            println("  Matrix $dims_str ($ct_label, $(n_elements) slots / batch $(batch_size))")

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
                    r.multiplicative_depth,
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
