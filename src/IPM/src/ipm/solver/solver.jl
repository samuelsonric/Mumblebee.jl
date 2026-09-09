abstract type AbstractSolver{T} end

const FORCING_FRAC = 0.1
const FORCING_CEIL = 0.3

include("ipm.jl")
include("utils.jl")

function isoptimal(s::AbstractSolver, μ, μs, pobj, dobj, pres, dres)
    return isoptimal(μ, μs, s.μ, pobj, dobj, pres, dres, s.ν;
        gap_tol=s.settings.gap_tol, feas_tol=s.settings.feas_tol)
end

function isnearoptimal(s::AbstractSolver, μ, μs, pobj, dobj, pres, dres)
    f = s.settings.near_factor
    return isoptimal(μ, μs, s.μ, pobj, dobj, pres, dres, s.ν;
        gap_tol=f * s.settings.gap_tol, feas_tol=f * s.settings.feas_tol)
end

function isoptimal(μ::T, μs::T, μt::T, pobj::T, dobj::T, pres::T, dres::T, ν::Integer; gap_tol::T, feas_tol::T) where {T}
    pres < feas_tol && dres < feas_tol || return false

    tol = gap_tol * max(one(T), min(abs(pobj), abs(dobj)))

    if μt > 0 && ν > 0
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

function getaug(s::AbstractSolver{T}) where {T}
    nH = norm(s.H)

    if iszero(s.nB[]) || iszero(nH)
        δ = one(T)
    elseif isempty(s.hist)
        δ = s.settings.aug_tol * s.nB[]^2 / nH
    else
        δ = getaug(s.hist)
    end

    return δ
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

function initkkt!(s::AbstractSolver, δ)
    return initkkt!(s.kkt, s.H; δ)
end

############################################################################################
# maxsteps
############################################################################################

function maxsteps(s::AbstractSolver, Δp::AbstractVector, Δd::AbstractVector, step_frac::Real)
    return maxsteps(s.sched, s.K, s.p, s.d, Δp, Δd, s.caches, s.B, s.wrk.step, step_frac)
end

############################################################################################
# scale!
############################################################################################

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

############################################################################################
# differential fan-out wrappers (over the cone.jl block fan-outs)
############################################################################################

function dualshadow!(sd::AbstractVector, s::IPMSolver)
    return dualshadow!(sd, s.B, s.p, s.K, s.caches, s.sched)
end

function primalhess!(r::AbstractVector, s::IPMSolver, Δp::AbstractVector)
    return primalhess!(r, s.B, s.p, s.K, s.caches, s.sched, Δp)
end

function primalthird!(r::AbstractVector, s::IPMSolver, Δp1::AbstractVector, Δp2::AbstractVector)
    return primalthird!(r, s.B, s.p, s.K, s.caches, s.sched, Δp1, Δp2)
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
