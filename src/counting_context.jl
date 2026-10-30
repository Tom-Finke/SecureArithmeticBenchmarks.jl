"""
    OpCounters

Mutable counters for OpenFHE primitive operations. Tracks call count and
cumulative wall-clock nanoseconds for EvalAdd, EvalSub, EvalMult, EvalRotate,
and EvalNegate.

!!! note "Thread safety"
    Counters use plain integers — not atomic. Disable Julia threading and set
    `OMP_NUM_THREADS=1` before benchmarking for accurate results.
"""
mutable struct OpCounters
    eval_add_count::Int
    eval_add_time_ns::Int
    eval_sub_count::Int
    eval_sub_time_ns::Int
    eval_mult_count::Int
    eval_mult_time_ns::Int
    eval_rotate_count::Int
    eval_rotate_time_ns::Int
    eval_negate_count::Int
    eval_negate_time_ns::Int
    resize_count::Int
    resize_time_ns::Int
end

OpCounters() = OpCounters(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

function reset!(c::OpCounters)
    c.eval_add_count = 0;    c.eval_add_time_ns = 0
    c.eval_sub_count = 0;    c.eval_sub_time_ns = 0
    c.eval_mult_count = 0;   c.eval_mult_time_ns = 0
    c.eval_rotate_count = 0; c.eval_rotate_time_ns = 0
    c.eval_negate_count = 0; c.eval_negate_time_ns = 0
    c.resize_count = 0;      c.resize_time_ns = 0
    nothing
end

function get_counters(c::OpCounters)
    (
        eval_add    = (count = c.eval_add_count,    time_s = c.eval_add_time_ns / 1e9),
        eval_sub    = (count = c.eval_sub_count,    time_s = c.eval_sub_time_ns / 1e9),
        eval_mult   = (count = c.eval_mult_count,   time_s = c.eval_mult_time_ns / 1e9),
        eval_rotate = (count = c.eval_rotate_count, time_s = c.eval_rotate_time_ns / 1e9),
        eval_negate = (count = c.eval_negate_count, time_s = c.eval_negate_time_ns / 1e9),
        resize      = (count = c.resize_count,      time_s = c.resize_time_ns / 1e9),
    )
end

"""
    CountingCryptoContext{T}

Transparent wrapper around an OpenFHE `CryptoContext` (`SharedPtr`) that counts
and times every `EvalAdd`, `EvalSub`, `EvalMult`, `EvalRotate`, and `EvalNegate`
call routed through it.

All other OpenFHE operations (key generation, encoding, encrypt/decrypt, …) are
forwarded unchanged to the inner context.

## Usage

    cc = GenCryptoContext(parameters)
    # … Enable, EvalBootstrapSetup on raw cc …
    counters = OpCounters()
    ccc = CountingCryptoContext(cc, counters)
    context = SecureContext(OpenFHEBackend(ccc))
    # … use context normally; counters accumulate …
"""
struct CountingCryptoContext{T}
    cc::T
    counters::OpCounters
end

"""
    make_counting_context(cc) -> (CountingCryptoContext, OpCounters)

Convenience constructor: wraps `cc` and returns both the wrapper and its counter
object.
"""
function make_counting_context(cc)
    counters = OpCounters()
    (CountingCryptoContext(cc, counters), counters)
end

# ---------------------------------------------------------------------------
# Counted operations — intercept via type piracy on CountingCryptoContext
# ---------------------------------------------------------------------------

function OpenFHE.EvalAdd(ctx::CountingCryptoContext, args...; kwargs...)
    t = time_ns()
    r = OpenFHE.EvalAdd(ctx.cc, args...; kwargs...)
    ctx.counters.eval_add_time_ns += time_ns() - t
    ctx.counters.eval_add_count += 1
    r
end

function OpenFHE.EvalSub(ctx::CountingCryptoContext, args...; kwargs...)
    t = time_ns()
    r = OpenFHE.EvalSub(ctx.cc, args...; kwargs...)
    ctx.counters.eval_sub_time_ns += time_ns() - t
    ctx.counters.eval_sub_count += 1
    r
end

function OpenFHE.EvalMult(ctx::CountingCryptoContext, args...; kwargs...)
    t = time_ns()
    r = OpenFHE.EvalMult(ctx.cc, args...; kwargs...)
    ctx.counters.eval_mult_time_ns += time_ns() - t
    ctx.counters.eval_mult_count += 1
    r
end

function OpenFHE.EvalRotate(ctx::CountingCryptoContext, args...; kwargs...)
    t = time_ns()
    r = OpenFHE.EvalRotate(ctx.cc, args...; kwargs...)
    ctx.counters.eval_rotate_time_ns += time_ns() - t
    ctx.counters.eval_rotate_count += 1
    r
end

function OpenFHE.EvalNegate(ctx::CountingCryptoContext, args...; kwargs...)
    t = time_ns()
    r = OpenFHE.EvalNegate(ctx.cc, args...; kwargs...)
    ctx.counters.eval_negate_time_ns += time_ns() - t
    ctx.counters.eval_negate_count += 1
    r
end

# ---------------------------------------------------------------------------
# Forwarded operations — pass through to the inner CryptoContext unchanged
# ---------------------------------------------------------------------------

OpenFHE.KeyGen(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.KeyGen(ctx.cc, args...; kwargs...)

OpenFHE.EvalMultKeyGen(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.EvalMultKeyGen(ctx.cc, args...; kwargs...)

OpenFHE.EvalRotateKeyGen(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.EvalRotateKeyGen(ctx.cc, args...; kwargs...)

OpenFHE.EvalBootstrapKeyGen(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.EvalBootstrapKeyGen(ctx.cc, args...; kwargs...)

OpenFHE.EvalBootstrap(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.EvalBootstrap(ctx.cc, args...; kwargs...)

OpenFHE.EvalBootstrapPrecompute(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.EvalBootstrapPrecompute(ctx.cc, args...; kwargs...)

OpenFHE.GetEncodingParams(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.GetEncodingParams(ctx.cc, args...; kwargs...)

OpenFHE.MakeCKKSPackedPlaintext(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.MakeCKKSPackedPlaintext(ctx.cc, args...; kwargs...)

OpenFHE.Encrypt(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.Encrypt(ctx.cc, args...; kwargs...)

OpenFHE.Decrypt(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.Decrypt(ctx.cc, args...; kwargs...)

OpenFHE.Enable(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.Enable(ctx.cc, args...; kwargs...)

OpenFHE.EvalBootstrapSetup(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.EvalBootstrapSetup(ctx.cc, args...; kwargs...)

OpenFHE.SerializeToString(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.SerializeToString(ctx.cc, args...; kwargs...)

OpenFHE.SerializeEvalMultKeyToString(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.SerializeEvalMultKeyToString(ctx.cc, args...; kwargs...)

OpenFHE.SerializeEvalAutomorphismKeyToString(ctx::CountingCryptoContext, args...; kwargs...) =
    OpenFHE.SerializeEvalAutomorphismKeyToString(ctx.cc, args...; kwargs...)

const CountingOpenFHEBackend = SecureArithmetic.OpenFHEBackend{<:CountingCryptoContext}
const CountingVec = Union{SecureArithmetic.PlainVector{<:CountingOpenFHEBackend},
                          SecureArithmetic.SecureVector{<:CountingOpenFHEBackend}}
const OrigResizeTypes = Tuple{Union{SecureArithmetic.PlainVector{<:SecureArithmetic.OpenFHEBackend},
                                    SecureArithmetic.SecureVector{<:SecureArithmetic.OpenFHEBackend}},
                              Integer}

function SecureArithmetic.resize(a::CountingVec, n::Integer)
    cc = SecureArithmetic.get_crypto_context(a.context)
    t = time_ns()
    result = invoke(SecureArithmetic.resize, OrigResizeTypes, a, n)
    cc.counters.resize_time_ns += time_ns() - t
    cc.counters.resize_count += 1
    result
end
