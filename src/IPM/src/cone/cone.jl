const SMALL_CONE_THRESHOLD = 3

"""
    AbstractCone

A convex cone.
"""
abstract type AbstractCone end

abstract type AbstractCache{C <: AbstractCone} end

struct Caches{T, I}
    #
    # The ith cache corresponds to the columns
    #
    #   xcol[i] ... xcol[i + 1] - 1
    #
    xcol::FVector{I}
    #
    # The ith cache corresponds to the slots
    #
    #   xblk[i] ... xblk[i + 1] - 1
    #
    xblk::FVector{I}
    #
    # The value
    #
    #   val(b)
    #
    # at slot b.
    #
    val::FVector{T}
end

struct ConeWorkspace{T}
    data::FVector{T}
end

struct ConeSchedule{T, I}
    nsmll::I
    xsmll::FVector{I}
    small::FVector{ConeWorkspace{T}}
    large::ConeWorkspace{T}
end

function Caches(cones::AbstractVector, B::BlockSparseMatrix{T, I}) where {T, I}
    xcol = FVector{I}(undef, nvtxs(B) + one(I))
    xblk = FVector{I}(undef, nvtxs(B) + one(I))

    c = zero(I)
    b = zero(I)

    for v in vtxs(B)
        ncol = ncols(B, v)
        xcol[v] = c + one(I); c += ncol
        xblk[v] = b + one(I); b += cachesize(cones[v], ncol)
    end

    val = FVector{T}(undef, b)

    xcol[nvtxs(B) + one(I)] = c + one(I)
    xblk[nvtxs(B) + one(I)] = b + one(I)

    return Caches(xcol, xblk, val)
end

function ConeWorkspace{T}(m::Integer) where {T}
    data = FVector{T}(undef, m)
    return ConeWorkspace{T}(data)
end

function ConeSchedule{T}(cones::AbstractVector, B::BlockSparseMatrix{T, I}, tdmax::I) where {T, I}
    rsmll = zero(I)
    rlarg = zero(I)

    nsmll = zero(I)
    tsmll = zero(I)

    for v in vtxs(B)
        ncol = ncols(B, v)

        rlarg = max(rlarg, workspacesize(cones[v], ncol))

        if ncol <= SMALL_CONE_THRESHOLD
            rsmll = max(rsmll, workspacesize(cones[v], ncol))

            nsmll += one(I)
            tsmll += ncol
        end
    end

    nsmll = min(nsmll, tdmax)

    xsmll = FVector{I}(undef, nsmll + one(I))
    small = FVector{ConeWorkspace{T}}(undef, nsmll)
    large = ConeWorkspace{T}(rlarg)

    v = zero(I)
    Δ = zero(I)

    xsmll[one(I)] = one(I)

    for s in oneto(nsmll)
        small[s] = ConeWorkspace{T}(rsmll)

        while v < nvtxs(B) && nsmll * Δ < tsmll
            v += one(I); ncol = ncols(B, v)

            if ncol ≤ SMALL_CONE_THRESHOLD
                Δ += ncol
            end
        end

        xsmll[s + one(I)] = v + one(I)
        Δ = zero(I)
    end

    return ConeSchedule{T, I}(nsmll, xsmll, small, large)
end

"""
    degree(cone::AbstractCone, n::Integer)

Get the rank of a cone with embedding
dimension `n`.
"""
degree(cone::AbstractCone)

"""
    identity!(x::AbstractVector, cone::AbstractCone)

Set x to the fixed point -f'(e) = e of the barrier.
"""
identity!(x::AbstractVector, cone::AbstractCone)

"""
    scale!(H, p, d, cache, work)

Set H to the Tuncel scaling matrix. If p
and d are elements of a symmetric cone, this
is the Hessian f''(w) of the barrier at the
Nesterov-Todd scaling point w.
"""
scale!(H::AbstractMatrix, p::AbstractVector, d::AbstractVector, cache::AbstractCache, work::ConeWorkspace)

function scale!(
        cone::AbstractCone,
        v::Integer,
        H::BlockSparseMatrix,
        Q::BlockSparseMatrix,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        B::BlockSparseMatrix,
        conewrk::ConeWorkspace,
    )
    rv = colrange(B, v)
    cv = cache(caches, v, cone)

    for e in srcrange(H, v)
        if H.tgt[e] == v
            Hv = block(H, v, v, e)
            Qv = block(Q, v, v, e)

            pv = view(p, rv)
            dv = view(d, rv)

            copyto!(Hv, Qv)
            return scale!(Hv, pv, dv, cv, conewrk)
        end
    end

    error()
end

function scale!(
        sched::ConeSchedule,
        K::AbstractVector,
        H::BlockSparseMatrix,
        Q::BlockSparseMatrix,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        B::BlockSparseMatrix,
        flags::AbstractVector,
        spsds::AbstractVector,
    )
    if sched.nsmll <= 1
        scale_st!(sched, K, H, Q, caches, p, d, B, flags, spsds)
    else
        scale_mt!(sched, K, H, Q, caches, p, d, B, flags, spsds)
    end
    return all(flags), sum(spsds)
end

function scale_st!(
        sched::ConeSchedule,
        K::AbstractVector,
        H::BlockSparseMatrix,
        Q::BlockSparseMatrix,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        B::BlockSparseMatrix,
        flags::AbstractVector,
        spsds::AbstractVector,
    )
    @inbounds for v in vtxs(B)
        flags[v], spsds[v] = scale!(K[v], v, H, Q, caches, p, d, B, sched.large)
    end

    return
end

function scale_mt!(
        sched::ConeSchedule{<:Any, I},
        K::AbstractVector,
        H::BlockSparseMatrix,
        Q::BlockSparseMatrix,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        B::BlockSparseMatrix,
        flags::AbstractVector,
        spsds::AbstractVector,
    ) where {I}
    @inbounds for v in vtxs(B)
        if ncols(B, v) > SMALL_CONE_THRESHOLD
            flags[v], spsds[v] = scale!(K[v], v, H, Q, caches, p, d, B, sched.large)
        end
    end

    @threads for s in oneto(sched.nsmll)
        ws = sched.small[s]

        sstrt = sched.xsmll[s]
        sstop = sched.xsmll[s + one(I)] - one(I)

        @inbounds for v in sstrt:sstop
            if ncols(B, v) <= SMALL_CONE_THRESHOLD
                flags[v], spsds[v] = scale!(K[v], v, H, Q, caches, p, d, B, ws)
            end
        end
    end

    return
end

"""
    corr!(r, p, d, Δp, Δd, σμ, cache, work)

Set r to the Mehrotra corrector term
r = -d - σμ f'(p) - η, where η is the third-order
correction η = -½ f'''(p)[Δp, f''(p)⁻¹ Δd]. If
p and d are elements of a symmetric cone, this
formula simplifies to r = -d + (σμ e - Δp ∘ Δd) / p,
where Δp ∘ Δd is the Jordan product of Δp and Δd.
"""
corr!(r::AbstractVector, p::AbstractVector, d::AbstractVector, Δp::AbstractVector, Δd::AbstractVector, σμ::Number, cache::AbstractCache, work::ConeWorkspace)

function initcorrector!(
        cone::AbstractCone,
        v::Integer,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
        conewrk::ConeWorkspace,
    )
    r = colrange(B, v)
    cv = cache(caches, v, cone)
    corr!(view(f, r), view(p, r), view(d, r), view(Δp, r), view(Δd, r), σμ, cv, conewrk)
    return
end

function initcorrector!(
        sched::ConeSchedule,
        K::AbstractVector,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
    )
    if sched.nsmll <= 1
        initcorrector_st!(sched, K, f, caches, p, d, Δp, Δd, σμ, B)
    else
        initcorrector_mt!(sched, K, f, caches, p, d, Δp, Δd, σμ, B)
    end

    return
end

function initcorrector_st!(
        sched::ConeSchedule,
        K::AbstractVector,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
    )
    @inbounds for v in vtxs(B)
        initcorrector!(K[v], v, f, caches, p, d, Δp, Δd, σμ, B, sched.large)
    end

    return
end

function initcorrector_mt!(
        sched::ConeSchedule{<:Any, I},
        K::AbstractVector,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
    ) where {I}
    @inbounds for v in vtxs(B)
        if ncols(B, v) > SMALL_CONE_THRESHOLD
            initcorrector!(K[v], v, f, caches, p, d, Δp, Δd, σμ, B, sched.large)
        end
    end

    @threads for s in oneto(sched.nsmll)
        ws = sched.small[s]

        sstrt = sched.xsmll[s]
        sstop = sched.xsmll[s + one(I)] - one(I)

        @inbounds for v in sstrt:sstop
            if ncols(B, v) <= SMALL_CONE_THRESHOLD
                initcorrector!(K[v], v, f, caches, p, d, Δp, Δd, σμ, B, ws)
            end
        end
    end

    return
end

"""
    corr0!(r, p, d, σμ, cache, work)

[`corr`](@ref) with Δp = Δd = 0.
"""
corr0!(r::AbstractVector, p::AbstractVector, d::AbstractVector, σμ::Number, cache::AbstractCache, work::ConeWorkspace)

function initpredictor!(
        cone::AbstractCone,
        v::Integer,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
        conewrk::ConeWorkspace,
    )
    r = colrange(B, v)
    cv = cache(caches, v, cone)
    corr0!(view(f, r), view(p, r), view(d, r), σμ, cv, conewrk)
    return
end

function initpredictor!(
        sched::ConeSchedule,
        K::AbstractVector,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
    )
    if sched.nsmll <= 1
        initpredictor_st!(sched, K, f, caches, p, d, σμ, B)
    else
        initpredictor_mt!(sched, K, f, caches, p, d, σμ, B)
    end

    return
end

function initpredictor_st!(
        sched::ConeSchedule,
        K::AbstractVector,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
    )
    @inbounds for v in vtxs(B)
        initpredictor!(K[v], v, f, caches, p, d, σμ, B, sched.large)
    end

    return
end

function initpredictor_mt!(
        sched::ConeSchedule{<:Any, I},
        K::AbstractVector,
        f::AbstractVector,
        caches::Caches,
        p::AbstractVector,
        d::AbstractVector,
        σμ::Real,
        B::BlockSparseMatrix,
    ) where {I}
    @inbounds for v in vtxs(B)
        if ncols(B, v) > SMALL_CONE_THRESHOLD
            initpredictor!(K[v], v, f, caches, p, d, σμ, B, sched.large)
        end
    end

    @threads for s in oneto(sched.nsmll)
        ws = sched.small[s]

        sstrt = sched.xsmll[s]
        sstop = sched.xsmll[s + one(I)] - one(I)

        @inbounds for v in sstrt:sstop
            if ncols(B, v) <= SMALL_CONE_THRESHOLD
                initpredictor!(K[v], v, f, caches, p, d, σμ, B, ws)
            end
        end
    end

    return
end

"""
    maxsteps(p, Δp, d, Δd, cache, work)

Compute the largest numbers 0 < τp, τd ≤ 1 such that
p + τp Δp and d + τd Δd lie in the interior of their
respective cones
"""
maxsteps(p::AbstractVector, Δp::AbstractVector, d::AbstractVector, Δd::AbstractVector, cache::AbstractCache, work::ConeWorkspace)

function maxsteps(
        cone::AbstractCone,
        v::Integer,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        caches::Caches,
        B::BlockSparseMatrix,
        conewrk::ConeWorkspace,
        step_frac::Real,
    )
    r = colrange(B, v)
    τp, τd = maxsteps(view(p, r), view(Δp, r), view(d, r), view(Δd, r), cache(caches, v, cone), conewrk)

    if !(cone isa CofreeCone)
        τp *= step_frac
        τd *= step_frac
    end

    return τp, τd
end

function maxsteps(
        sched::ConeSchedule,
        K::AbstractVector,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        caches::Caches,
        B::BlockSparseMatrix,
        step::AbstractVector,
        step_frac::Real,
    )
    if sched.nsmll <= 1
        maxsteps_st(sched, K, p, d, Δp, Δd, caches, B, step, step_frac)
    else
        maxsteps_mt(sched, K, p, d, Δp, Δd, caches, B, step, step_frac)
    end

    return minimum(step)
end

function maxsteps_st(
        sched::ConeSchedule,
        K::AbstractVector,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        caches::Caches,
        B::BlockSparseMatrix,
        step::AbstractVector,
        step_frac::Real,
    )
    @inbounds for v in vtxs(B)
        a, b = maxsteps(K[v], v, p, d, Δp, Δd, caches, B, sched.large, step_frac)
        step[v] = min(a, b)
    end

    return
end

function maxsteps_mt(
        sched::ConeSchedule{<:Any, I},
        K::AbstractVector,
        p::AbstractVector,
        d::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        caches::Caches,
        B::BlockSparseMatrix,
        step::AbstractVector,
        step_frac::Real,
    ) where {I}
    @inbounds for v in vtxs(B)
        if ncols(B, v) > SMALL_CONE_THRESHOLD
            a, b = maxsteps(K[v], v, p, d, Δp, Δd, caches, B, sched.large, step_frac)
            step[v] = min(a, b)
        end
    end

    @threads for s in oneto(sched.nsmll)
        ws = sched.small[s]

        sstrt = sched.xsmll[s]
        sstop = sched.xsmll[s + one(I)] - one(I)

        @inbounds for v in sstrt:sstop
            if ncols(B, v) <= SMALL_CONE_THRESHOLD
                a, b = maxsteps(K[v], v, p, d, Δp, Δd, caches, B, ws, step_frac)
                step[v] = min(a, b)
            end
        end
    end

    return
end

"""
    dualshadow!(sd, p, cache, work)

Set sd to the dual "shadow" iterate d* = -f'(p).
"""
dualshadow!(sd::AbstractVector, p::AbstractVector, cache::AbstractCache, work::ConeWorkspace)

function dualshadow!(sd::AbstractVector, B::BlockSparseMatrix, p::AbstractVector, K::AbstractVector, caches::Caches, sched::ConeSchedule)
    for v in vtxs(B)
        r = colrange(B, v)
        dualshadow!(view(sd, r), view(p, r), cache(caches, v, K[v]), sched.large)
    end

    return sd
end

"""
    primalshadow!(sp, d, cache, work)

Set sp to the primal "shadow" iterate p*, solving -f'(p*) = d.
"""
primalshadow!(sp::AbstractVector, d::AbstractVector, cache::AbstractCache, work::ConeWorkspace)

"""
    primalhess!(r, p, Δp, cache, work)

Set r to the second derivative f''(p)[Δp]
"""
primalhess!(r::AbstractVector, p::AbstractVector, Δp::AbstractVector, cache::AbstractCache, work::ConeWorkspace)

function primalhess!(r::AbstractVector, B::BlockSparseMatrix, p::AbstractVector, K::AbstractVector, caches::Caches, sched::ConeSchedule, Δp::AbstractVector)
    for v in vtxs(B)
        rng = colrange(B, v)
        primalhess!(view(r, rng), view(p, rng), view(Δp, rng), cache(caches, v, K[v]), sched.large)
    end

    return r
end

"""
    primalthird!(r, p, Δp1, Δp2, cache, work)

Set r to the third derivative f'''(p)[Δp1, Δp2]
"""
primalthird!(r::AbstractVector, p::AbstractVector, Δp1::AbstractVector, Δp2::AbstractVector, cache::AbstractCache, work::ConeWorkspace)

function primalthird!(r::AbstractVector, B::BlockSparseMatrix, p::AbstractVector, K::AbstractVector, caches::Caches, sched::ConeSchedule, Δp1::AbstractVector, Δp2::AbstractVector)
    for v in vtxs(B)
        rng = colrange(B, v)
        primalthird!(view(r, rng), view(p, rng), view(Δp1, rng), view(Δp2, rng), cache(caches, v, K[v]), sched.large)
    end

    return r
end

"""
    cachesize(cone, n)

Return the number of cache slots needed for a cone
with embedding dimension n.
"""
function cachesize(cone::C, n::Integer) where {C <: AbstractCone}
    return cachesize(C, n)
end

"""
    workspacesize(cone, n)

Return the number of workspace floats needed for a cone
with embedding dimension n.
"""
function workspacesize(cone::C, n::Integer) where {C <: AbstractCone}
    return workspacesize(C, n)
end

function workspacesize(::Type{<:AbstractCone}, ::Integer)
    return 0
end

"""
    initcache!(cache)

Initialise a cache.
"""
function initcache!(c::AbstractCache)
    return c
end

"""
    cache(caches, i, cone)

Get the ith cache.
"""
cache(caches::Caches, i::Integer, cone::AbstractCone)

function cachedata(c::Caches, i::Integer)
    return view(c.val, c.xblk[i]:c.xblk[i + 1] - 1)
end

include("eig.jl")
include("sdp.jl")
include("tdc/tdc.jl")
include("pos.jl")
include("soc.jl")
include("noc.jl")
include("utils.jl")
