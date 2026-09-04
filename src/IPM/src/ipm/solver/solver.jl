abstract type AbstractSolver{T} end

const FORCING_FRAC = 0.1
const FORCING_CEIL = 0.3

include("ipm.jl")
include("utils.jl")

function isoptimal(s::AbstractSolver, μ, μs, pobj, dobj, pres, dres)
    return isoptimal(μ, μs, s.settings.relax_tol, pobj, dobj, pres, dres, s.ν;
        gap_tol=s.settings.gap_tol, feas_tol=s.settings.feas_tol)
end

function isnearoptimal(s::AbstractSolver, μ, μs, pobj, dobj, pres, dres)
    f = s.settings.near_factor
    return isoptimal(μ, μs, s.settings.relax_tol, pobj, dobj, pres, dres, s.ν;
        gap_tol=f * s.settings.gap_tol, feas_tol=f * s.settings.feas_tol)
end

function isoptimal(μ::T, μs::T, μt::T, pobj::T, dobj::T, pres::T, dres::T, ν::Integer; gap_tol::T, feas_tol::T) where {T}
    pres < feas_tol && dres < feas_tol || return false

    tol = gap_tol * max(one(T), min(abs(pobj), abs(dobj)))

    if μt > 0
        return ν * (μ - μt + μt * log1p(μs * μt - one(T))) ≤ tol
    else
        return pobj - dobj < tol
    end
end

function fnorm(Δf::AbstractVector, pscl::AbstractVector, sf::Real)
    return scalenorm(Δf, pscl) / (1 + sf)
end

function gnorm(Δg::AbstractVector, yscl::AbstractVector, sg::Real)
    return scalenorm(Δg, yscl) / (1 + sg)
end

function setaug!(s::AbstractSolver{T}) where {T}
    nH = norm(s.H)

    if iszero(s.nB[]) || iszero(nH)
        s.δ[] = one(T)
    elseif isempty(s.hist)
        s.δ[] = s.settings.aug_tol * s.nB[]^2 / nH
    else
        s.δ[] = getaug(s.hist)
    end

    return
end

function showsolver(io::IO, s::AbstractSolver; indent::Integer=0)
    return showsettings(io, s.settings; indent)
end

function Base.show(io::IO, ::MIME"text/plain", s::T) where {T <: AbstractSolver}
    println(io, T, ":")
    return showsolver(io, s; indent=2)
end

function print_timers(s::AbstractSolver)
    display(s.timers)
end

function isstalled(s::AbstractSolver)
    return isstalled(s.hist, s.settings.stall_tol)
end

function initkkt!(s::AbstractSolver{T}) where {T}
    return initkkt!(s.kkt, s.H; δ=s.δ[])
end

############################################################################################
# maxsteps
############################################################################################

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

function maxsteps(s::AbstractSolver, Δp::AbstractVector, Δd::AbstractVector, step_frac::Real)
    return maxsteps(s.sched, s.K, s.p, s.d, Δp, Δd, s.caches, s.B, s.wrk.step, step_frac)
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

############################################################################################
# scale!
############################################################################################

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

function scale!(::CofreeCone, ::Integer, ::BlockSparseMatrix{T}, ::BlockSparseMatrix, ::Caches, ::AbstractVector, ::AbstractVector, ::BlockSparseMatrix, ::ConeWorkspace) where {T}
    return true, zero(T)
end

#
# compute the Hessian
#
#   f''(w)
#
# of the primal barrier function f
# at the Nestorov-Todd scaling point w
#
# for non-symmetric cones, no such point
# exists, so the Hessian is replaced
# by a Tuncel scaling matrix
#
function scale!(s::AbstractSolver)
    return scale!(s.sched, s.K, s.H, s.Q, s.caches, s.p, s.d, s.B, s.wrk.flag, s.wrk.spsd)
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

############################################################################################
# initcorrector!
############################################################################################

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

############################################################################################
# initpredictor!
############################################################################################

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

function solve_impl!(s::AbstractSolver, io::IO)
    status = CONTINUE; i = 0

    if s.settings.verbose > 0
        showsettings(io, s.settings)
        println(io)
        showtop(io, s.hist)
    end

    while status == CONTINUE
        status = step!(s); i += 1

        if s.settings.verbose > 0
            showrow(io, i, s.hist[i])
        end

        if status != CONTINUE
            break
        end
    end

    if s.settings.verbose > 0
        showbot(io, s.hist)
    end

    return status
end

function CommonSolve.solve!(s::AbstractSolver)
    status = @timeit s.timers "solve" solve_impl!(s, stdout)
    return result(s, status)
end
