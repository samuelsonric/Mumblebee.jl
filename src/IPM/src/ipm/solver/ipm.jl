struct IPMSolver{T, I, V, KKT} <: AbstractSolver{T}
    Q::BlockSparseMatrix{T, I}
    H::BlockSparseMatrix{T, I}
    B::BlockSparseMatrix{T, I}
    f::FVector{T}
    g::FVector{T}
    p::FVector{T}
    d::FVector{T}
    y::FVector{T}
    K::FVector{V}
    scaling::IPMScaling{T}
    P2::FPermutation{I}
    P1::FPermutation{I}
    wrk::IPMWorkspace{T}
    caches::Caches{T, I}
    sched::ConeSchedule{T, I}
    kkt::KKT
    hist::IPMHistory{T}
    ν::Int
    settings::IPMSettings{T}
    nf::FScalar{T}
    ng::FScalar{T}
    sg::FScalar{T}     # ‖g‖ in original (unscaled) units — B-primal stopping-test denominator
    sf::FScalar{T}     # ‖f‖ in original (unscaled) units — dual stopping-test denominator
    nB::FScalar{T}      # ‖B‖ — fixed for the solver's lifetime; the cold-start augmentation anchor
    δ::FScalar{T}       # reciprocal augmentation 1/α; owned by setaug!
    timers::TimerOutput
end

function result(s::IPMSolver{T}, status::IPMStatus) where {T}
    pu = copy(s.p)
    du = copy(s.d)
    yu = copy(s.y)

    unscale!(pu, du, yu, s.scaling)   # user frame

    w = s.wrk; scl = s.scaling
    #
    # on an infeasibility status the certificate is the unscaled ray in p / y
    # (the caller rescales if desired); no objective is reported
    #
    if status in (PRIMAL_INFEASIBLE, NEAR_PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, NEAR_DUAL_INFEASIBLE) || isempty(s.hist)
        mu = pres = dres = pobj = dobj = T(NaN)
    else
        residuals!(s)
        pQp  = dot(s.p, s.Q, s.p)
        pobj = pQp / 2 - dot(s.f, s.p)
        dobj = dot(s.g, s.y) - pQp / 2
        mu   = iszero(s.ν) ? T(NaN) : dot(s.p, s.d) / s.ν
        pres = gnorm(w.Δg, scl.yscl, s.sg[])
        dres = fnorm(w.Δf, scl.pscl, s.sf[])
    end

    p = Vector{T}(undef, length(s.p))
    d = Vector{T}(undef, length(s.d))

    ldiv!(p, s.P2, pu)
    ldiv!(d, s.P2, du)
    #
    # the B-row dual un-permutes through P1 into user space
    #
    y = s.P1 \ yu

    niter = 0
    nsolve = 0

    for row in s.hist
        niter += 1
        nsolve += row.piter + row.ppass + row.citer + row.cpass
    end

    return IPMResult{T}(p, d, y, status, niter, nsolve, s.hist, s.timers,
                        mu, pres, dres, pobj, dobj)
end

############################################################################################
# residuals!
############################################################################################

#
# compute negated residuals
#
#   [ Δf ]   [ d + f ]   [  Q  -Bᵀ ] [ p ]
#   [ Δg ] = [   g   ] - [  B   0  ] [ y ]
#
function residuals!(s::IPMSolver{T}) where {T}
    w = s.wrk
    mulkkt!(w.Δf, w.Δg, s.Q, s.B, s.p, s.y)

    @inbounds for i in eachindex(w.Δf, s.d, s.f)
        w.Δf[i] = s.d[i] + s.f[i] - w.Δf[i]
    end

    @inbounds for i in eachindex(w.Δg, s.g)
        w.Δg[i] = s.g[i] - w.Δg[i]
    end

    return w.Δf, w.Δg
end

############################################################################################
# solvepredictor! / solvecorrector!
############################################################################################

#
# solve for the Mehrotra predictor direction
#
#   [ H  -Bᵀ ] [ Δpa ]   [ Δf - d ]
#   [ B   0  ] [ Δya ] = [ Δg     ]
#
function solvepredictor!(s::IPMSolver{T}; ftol::T, gtol::T) where {T}
    return solvepredictor!(
        s.wrk, s.kkt, s.settings, s.H, s.B, s.Q, s.K, s.p, s.d,
        s.caches, s.sched, s.timers;
        ftol, gtol,
    )
end

function solvepredictor!(
        w::IPMWorkspace{T},
        kkt::KKTSolver{T},
        set::IPMSettings{T},
        H::BlockSparseMatrix{T},
        B::BlockSparseMatrix{T},
        Q::BlockSparseMatrix{T},
        K::AbstractVector,
        p::AbstractVector{T},
        d::AbstractVector{T},
        caches::Caches{T},
        sched::ConeSchedule{T},
        timers::TimerOutput;
        ftol::T,
        gtol::T,
    ) where {T}
    if set.relax_tol > 0
        @timeit timers "init" initpredictor!(sched, K, w.f, caches, p, d, set.relax_tol, B)
    else
        axpby!(-1, d, 0, w.f)
    end

    axpy!(1, w.Δf, w.f)
    #
    # solve for the directions Δpa, Δya (base + internal refinement)
    #
    #   [ H  -Bᵀ ] [ Δpa ]   [ Δf - d ]
    #   [ B   0  ] [ Δya ] = [ Δg     ]
    #
    piter, ppass, pstat, dmin, dmax = @timeit timers "solve" solvekkt!(
        kkt, w.Δpa, w.Δya, H, B, w.f, w.Δg;
        warm=false, ftol, gtol, stall=set.refine_stall_tol, irmax=set.refine_max_iter, cgmax=set.newton_max_iter, irmin=1,
    )
    #
    # recover Δda:
    #
    #   Δda ← Q Δpa - Bᵀ Δya - Δf
    #
    copyto!(w.Δda, w.Δf)
    mul!(w.Δda, B', w.Δya, -1, -1)
    mul!(w.Δda, Q, w.Δpa, 1, 1)

    return piter, ppass, pstat, dmin, dmax
end

#
# solve for the Mehrotra combined direction
#
#   [ H  -Bᵀ ] [ Δp ]   [ Δf* ]
#   [ B   0  ] [ Δy ] = [ Δg  ]
#
# where Δf* is the corrected dual residual
#
function solvecorrector!(s::IPMSolver{T}, μ::T; ftol::T, gtol::T) where {T}
    return solvecorrector!(
        s.wrk, s.kkt, s.settings, s.H, s.B, s.Q, s.K, s.p, s.d,
        s.caches, s.sched, s.ν, μ, s.timers;
        ftol, gtol,
    )
end

function solvecorrector!(
        w::IPMWorkspace{T},
        kkt::KKTSolver{T},
        set::IPMSettings{T},
        H::BlockSparseMatrix{T},
        B::BlockSparseMatrix{T},
        Q::BlockSparseMatrix{T},
        K::AbstractVector,
        p::AbstractVector{T},
        d::AbstractVector{T},
        caches::Caches{T},
        sched::ConeSchedule{T},
        ν::Integer,
        μ::T,
        timers::TimerOutput;
        ftol::T,
        gtol::T,
    ) where {T}
    μt = set.relax_tol
    #
    # compute the largest step length τa ∈ (0, 1]
    # such that the perturbed iterates
    #
    #   p + τa Δpa ∈ K
    #   d + τa Δda ∈ K*
    #
    # lie within their respective cones
    #
    τa = @timeit timers "maxsteps" maxsteps(sched, K, p, d, w.Δpa, w.Δda, caches, B, w.step, one(T))
    #
    # compute the centering parameter
    #
    #   σμ ← clamp(μa (μa / μ)², 0, μ)
    #
    # where
    #
    #   μa  = ⟨p + τa Δpa, d + τa Δda⟩ / ν
    #
    σμ = zero(T)

    for j in cols(B)
        σμ += (p[j] + τa * w.Δpa[j]) * (d[j] + τa * w.Δda[j])
    end

    μa = σμ / ν

    if μt > 0
        Δμ = μ - μt

        if Δμ > 0
            Δμa = clamp(μa - μt, zero(T), Δμ)
            σμ = μt + clamp(Δμa * (Δμa / Δμ)^2, zero(T), Δμ)
        else
            σμ = μt           # at or below target ⇒ pure re-centering at μ′
        end
    else
        σμ = clamp(μa * (μa / μ)^2, zero(T), μ)
    end
    #
    # set f to the Mehrota corrector term:
    #
    #   f ← -d + (σμ e - Δpa ∘ Δda) / p
    #
    # where e is the Jordan identity element e ∈ K.
    #
    @timeit timers "init" initcorrector!(sched, K, w.f, caches, p, d, w.Δpa, w.Δda, σμ, B)

    axpy!(1, w.Δf, w.f)
    #
    # solve for the directions Δp, Δy
    #
    #   [ H  -Bᵀ ] [ Δp ]   [ Δf* ]
    #   [ B   0  ] [ Δy ] = [ Δg  ]
    #
    copyto!(w.Δp, w.Δpa)
    copyto!(w.Δy, w.Δya)

    citer, cpass, cstat, _, _ = @timeit timers "solve" solvekkt!(
        kkt, w.Δp, w.Δy, H, B, w.f, w.Δg;
        warm=true, ftol, gtol, stall=set.refine_stall_tol, irmax=set.refine_max_iter, cgmax=set.newton_max_iter, irmin=1,
    )
    #
    # recover Δd:
    #
    #   Δd ← Q Δp - Bᵀ Δy - Δf
    #
    copyto!(w.Δd, w.Δf)
    mul!(w.Δd, B', w.Δy, -1, -1)
    mul!(w.Δd, Q, w.Δp, 1, 1)

    return citer, cpass, cstat
end

############################################################################################
# solveexact!
############################################################################################

function solveexact!(s::IPMSolver{T}) where {T}
    return solveexact!(
        s.wrk, s.kkt, s.settings, s.H, s.B, s.scaling, s.sf[], s.sg[], s.timers,
    )
end

function solveexact!(
        w::IPMWorkspace{T},
        kkt::KKTSolver{T},
        set::IPMSettings{T},
        H::BlockSparseMatrix{T},
        B::BlockSparseMatrix{T},
        scaling::IPMScaling{T},
        sf::T,
        sg::T,
        timers::TimerOutput,
    ) where {T}
    #
    # solve the KKT system
    #
    #   [ H  -Bᵀ ] [ Δpa ]   [ Δf ]
    #   [ B   0  ] [ Δya ] = [ Δg ]
    #
    function fstop(δf)
        return fnorm(δf, scaling.pscl, sf) ≤ set.feas_tol
    end

    function gstop(δg)
        return gnorm(δg, scaling.yscl, sg) ≤ set.feas_tol
    end

    piter, ppass, pstat, dmin, dmax = @timeit timers "solve" solvekkt!(
        fstop, gstop, kkt, w.Δpa, w.Δya, H, B, w.Δf, w.Δg;
        warm=false, ftol=eps(T), gtol=eps(T), stall=set.refine_stall_tol, irmax=set.refine_max_iter, cgmax=set.newton_max_iter,
    )

    return piter, ppass, pstat, dmin, dmax
end

############################################################################################
# identitypoint! / startingpoint!
############################################################################################

function identitypoint!(p::AbstractVector{T}, d::AbstractVector{T}, B::BlockSparseMatrix{T}, K) where {T}
    for v in vtxs(B)
        r = colrange(B, v)
        identity!(view(p, r), K[v])
        identity!(view(d, r), K[v])
    end

    return p, d
end

function startingpoint!(
        p::AbstractVector{T},
        d::AbstractVector{T},
        B::BlockSparseMatrix{T},
        Q::BlockSparseMatrix{T},
        g::AbstractVector{T},
        f::AbstractVector{T},
        K::AbstractVector,
    ) where {T}
    identitypoint!(p, d, B, K)

    z = B * p
    w = Q * p

    np = norm(p)
    nz = norm(z)
    nw = norm(w)

    if nz > eps(T) * np
        sp = max(one(T), norm(g) / nz)
    else
        sp = one(T)
    end

    if np > eps(T)
        sd = max(one(T), (norm(f) + sp * nw) / np)
    else
        sd = one(T)
    end

    lmul!(sp, p)
    lmul!(sd, d)

    return p, d
end

############################################################################################
# constructor / reinit! / init
############################################################################################

function reinit!(s::IPMSolver; p0=nothing, d0=nothing, y0=nothing, f=nothing, g=nothing)
    return reinit!(s, p0, d0, y0, f, g)
end

function reinit!(s::IPMSolver, p0, d0, y0, f, g)
    if !isnothing(f)
        mul!(s.f, s.P2, f)
        s.f .*= s.scaling.pscl
        s.nf[] = norm(s.f)
        s.sf[] = scalenorm(s.f, s.scaling.pscl)
    end

    if !isnothing(g)
        mul!(s.g, s.P1, g)
        s.g .*= s.scaling.yscl
        s.ng[] = norm(s.g)
        s.sg[] = scalenorm(s.g, s.scaling.yscl)
    end

    if isnothing(p0) && isnothing(d0)
        startingpoint!(s.p, s.d, s.B, s.Q, s.g, s.f, s.K)

        if isnothing(y0)
            fill!(s.y, false)
        else
            mul!(s.y, s.P1, y0)
            s.y ./= s.scaling.yscl
        end
    else
        isnothing(p0) || mul!(s.p, s.P2, p0)
        isnothing(d0) || mul!(s.d, s.P2, d0)

        for v in vtxs(s.B)
            r = colrange(s.B, v)

            if isnothing(d0)
                dualshadow!(view(s.d, r), view(s.p, r), cache(s.caches, v, s.K[v]), s.sched.large)
            elseif isnothing(p0)
                primalshadow!(view(s.p, r), view(s.d, r), cache(s.caches, v, s.K[v]), s.sched.large)
            end
        end

        if isnothing(y0)
            fill!(s.y, false)
        else
            mul!(s.y, s.P1, y0)
        end

        scale!(s.p, s.d, s.y, s.scaling)
    end

    for v in vtxs(s.B)
        initcache!(cache(s.caches, v, s.K[v]))
    end

    empty!(s.hist)

    return s
end

function IPMSolver(prob::IPMProblem{T, I}, settings::IPMSettings{T}; p0=nothing, d0=nothing, y0=nothing) where {T, I}
    n = size(prob.B, 2)
    m = size(prob.B, 1)
    ν = conedegree(prob.K, prob.B)

    S, Q, B, f, g, cones, P1, P2 = symbkkt(prob, settings.elim_alg)

    scaling = IPMScaling{T}(n, m)

    if settings.scale_max_iter > 0
        equilibrate!(scaling, B, Q, f, g; itmax=settings.scale_max_iter)
    end

    if settings.pivot
        kkt = PivotedUzawaSolver(S, B; cgmax=settings.newton_max_iter, irmax=settings.refine_max_iter)
    else
        kkt = UzawaSolver(S, B; cgmax=settings.newton_max_iter, irmax=settings.refine_max_iter)
    end

    p = FVector{T}(undef, n)
    d = FVector{T}(undef, n)
    y = FVector{T}(undef, m)

    caches = Caches(cones, B)

    H = copy(Q)
    sched = ConeSchedule{T}(cones, B, nthreads())
    ipmwrk = IPMWorkspace{T}(m, n, nvtxs(B))
    hist = IPMHistory{T}()
    nf = FScalar{T}(undef)
    ng = FScalar{T}(undef)
    sg = FScalar{T}(undef)
    sf = FScalar{T}(undef)
    nB = FScalar{T}(undef)
    δ = FScalar{T}(undef)

    nB[] = norm(B)
    nf[] = norm(f)
    ng[] = norm(g)
    sg[] = scalenorm(g, scaling.yscl)
    sf[] = scalenorm(f, scaling.pscl)

    solver = IPMSolver(Q, H, B, f, g, p, d, y, cones,
        scaling, P2, P1, ipmwrk, caches, sched, kkt,
        hist, ν, settings, nf, ng, sg, sf, nB, δ, TimerOutput()
    )

    return reinit!(solver; p0, d0, y0)
end

function IPMSolver(prob::IPMProblem{T}; p0=nothing, d0=nothing, y0=nothing, kw...) where {T}
    settings = IPMSettings{T}(; kw...)
    return IPMSolver(prob, settings; p0, d0, y0)
end

function CommonSolve.init(prob::IPMProblem{T}, settings::IPMSettings{T}; p0=nothing, d0=nothing, y0=nothing) where {T}
    return IPMSolver(prob, settings; p0, d0, y0)
end

function CommonSolve.init(prob::IPMProblem{T}; kw...) where {T}
    return IPMSolver(prob; kw...)
end

############################################################################################
# infeasibility certificates
############################################################################################
#
# The affine iterate is asymptotically a certificate (Todd, FoCM 2004): on a
# primal-infeasible problem ‖y‖ → ∞ with Qp - d - Bᵀy = O(1), so y/‖y‖ certifies
# ∃y: -Bᵀy ∈ K*, gᵀy > 0. These are the HSD tests with the τ/κ gate dropped
# (the affine solver is the τ = 1 slice).
#
function isprimalinfeasible(s::IPMSolver, rtol, atol)
    w = s.wrk
    gy = dot(s.g, s.y)
    ny = norm(s.y)
    flag = gy > atol * ny * (1 + s.ng[])

    if flag
        nQp = norm(w.Qp)
        copyto!(w.f, w.Qp)
        axpy!(-1, s.d, w.f)
        mul!(w.f, s.B', s.y, -1, 1)          # Qp - d - Bᵀy
        flag = max(nQp, norm(w.f)) < rtol * gy * (1 + norm(s.d) / ny)
    end

    return flag
end

function isprimalinfeasible(s::IPMSolver)
    return isprimalinfeasible(s, s.settings.infeas_rel, s.settings.infeas_abs)
end

function isnearprimalinfeasible(s::IPMSolver)
    f = s.settings.near_factor
    return isprimalinfeasible(s, f * s.settings.infeas_rel, s.settings.infeas_abs)
end

function isdualinfeasible(s::IPMSolver, rtol, atol)
    w = s.wrk
    fp = dot(s.f, s.p)
    np = norm(s.p)
    flag = fp > atol * np * (1 + s.nf[])

    if flag
        mul!(w.Δya, s.B, s.p)   # Δya is free scratch here (the predictor overwrites it next)
        flag = max(norm(w.Δya), norm(w.Qp)) < rtol * abs(fp)
    end

    return flag
end

function isdualinfeasible(s::IPMSolver)
    return isdualinfeasible(s, s.settings.infeas_rel, s.settings.infeas_abs)
end

function isneardualinfeasible(s::IPMSolver)
    f = s.settings.near_factor
    return isdualinfeasible(s, f * s.settings.infeas_rel, s.settings.infeas_abs)
end

function zerofree!(x::AbstractVector{T}, s::IPMSolver) where {T}
    for v in vtxs(s.B)
        if s.K[v] isa CofreeCone
            fill!(view(x, colrange(s.B, v)), zero(T))
        end
    end

    return x
end

function nearstatus(s::IPMSolver, status::IPMStatus, μ, μs, pobj, dobj, pres, dres)
    if isnearoptimal(s, μ, μs, pobj, dobj, pres, dres)
        status = NEAR_OPTIMAL
    elseif isnearprimalinfeasible(s)
        status = NEAR_PRIMAL_INFEASIBLE
    elseif isneardualinfeasible(s)
        status = NEAR_DUAL_INFEASIBLE
    end

    return status
end

############################################################################################
# step!
############################################################################################

function step!(s::IPMSolver)
    status = CONTINUE

    (; μ, step, pres, dres, pobj, dobj, ρ, piter, ppass, pstat, citer, cpass, cstat, dmin, dmax, χ) = defaultrow(s.hist)
    μt = s.settings.relax_tol   # barrier target μ′; 0 = exact solve

    w = s.wrk
    #
    # compute the inner product
    #
    #   pᵀd
    #
    pd = dot(s.p, s.d)
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
    flag, spsd = @timeit s.timers "scale" scale!(s)

    if !flag
        if s.settings.verbose > 1
            @warn "Scaling failed."
        end

        status = NUMERICAL_FAILURE
    else
        #
        # compute negated residuals
        #
        #   [ Δf ]   [ d + f ]   [  Q  -Bᵀ ] [ p ]
        #   [ Δg ] = [   g   ] - [  B   0  ] [ y ]
        #
        residuals!(s)
        #
        # compute the centrality parameter
        #
        #   μ = pᵀd / ν
        #
        μ = pd / max(s.ν, 1)
        #
        # compute the dual centrality parameter
        #
        #   μ* = p*ᵀ d* / ν
        #
        μs = spsd / max(s.ν, 1)
        #
        # compute the divergence parameter
        #
        #   χ = μ μ* - 1
        #
        # which measures how far the iterate (p, d, y)
        # is from the central path
        #
        χ = μ * μs - 1

        pres = gnorm(w.Δg, s.scaling.yscl, s.sg[])
        dres = fnorm(w.Δf, s.scaling.pscl, s.sf[])

        mul!(w.Qp, s.Q, s.p)
        pQp = dot(s.p, w.Qp)
        pobj = pQp / 2 - dot(s.f, s.p)
        dobj = dot(s.g, s.y) - pQp / 2

        if isoptimal(s, μ, μs, pobj, dobj, pres, dres)
            status = OPTIMAL
        elseif isprimalinfeasible(s)
            status = PRIMAL_INFEASIBLE
        elseif isdualinfeasible(s)
            status = DUAL_INFEASIBLE
        elseif length(s.hist) ≥ s.settings.max_iter
            status = nearstatus(s, ITERATION_LIMIT, μ, μs, pobj, dobj, pres, dres)
        elseif isstalled(s)
            status = nearstatus(s, STALLED, μ, μs, pobj, dobj, pres, dres)
        elseif iszero(s.ν)
            #
            # choose augmentation parameter δ
            #
            setaug!(s)

            initok, ρ = @timeit s.timers "initkkt" initkkt!(s)

            if !initok
                if s.settings.verbose > 1
                    @warn "Failed to initialize KKT solver."
                end

                status = nearstatus(s, NUMERICAL_FAILURE, μ, μs, pobj, dobj, pres, dres)
            else
                #
                # solve the KKT system
                #
                #   [ H  -Bᵀ ] [ Δpa ]   [ Δf ]
                #   [ B   0  ] [ Δya ] = [ Δg ]
                #
                piter, ppass, pstat, dmin, dmax = @timeit s.timers "exact" solveexact!(s)

                axpy!(1, w.Δpa, s.p)
                axpy!(1, w.Δya, s.y)
            end
        elseif !(μ > 0)
            if s.settings.verbose > 1
                @warn "Nonpositive μ."
            end

            status = nearstatus(s, NUMERICAL_FAILURE, μ, μs, pobj, dobj, pres, dres)
        else
            #
            # choose augmentation parameter δ
            #
            setaug!(s)

            initok, ρ = @timeit s.timers "initkkt" initkkt!(s)

            if !initok
                if s.settings.verbose > 1
                    @warn "Failed to initialize KKT solver."
                end

                status = nearstatus(s, NUMERICAL_FAILURE, μ, μs, pobj, dobj, pres, dres)
            else
                #
                # compute tolerances for predictor and corrector solves
                #
                #   ϵ = θ μ/μ₁
                #
                if isempty(s.hist.μ)
                    μ1 = μ
                else
                    μ1 = first(s.hist.μ)
                end

                if μt > 0
                    Δμ  = abs(μ  - μt)
                    Δμ1 = abs(μ1 - μt)
                    tol = FORCING_FRAC * min(Δμ / Δμ1, 1)
                else
                    tol = FORCING_FRAC * μ / μ1
                end

                ftol = tol * (1 + s.nf[])
                gtol = tol * (1 + s.ng[])
                #
                # solve for the Mehrotra predictor direction
                #
                #   [ H  -Bᵀ ] [ Δpa ]   [ Δf - d ]
                #   [ B   0  ] [ Δya ] = [ Δg     ]
                #
                piter, ppass, pstat, dmin, dmax = @timeit s.timers "predictor" solvepredictor!(s; ftol, gtol)

                zerofree!(w.Δda, s)
                #
                # solve for the Mehrotra combined direction
                #
                #   [ H  -Bᵀ ] [ Δp ]   [ Δf* ]
                #   [ B   0  ] [ Δy ] = [ Δg  ]
                #
                # where Δf* is the corrected dual residual
                #
                citer, cpass, cstat = @timeit s.timers "corrector" solvecorrector!(s, μ; ftol, gtol)

                zerofree!(w.Δd, s)

                if pstat !== KKT_SOLVED || cstat !== KKT_SOLVED
                    if s.settings.verbose > 1
                        @info "KKT solve above target tolerance" pstat cstat
                    end

                    status = nearstatus(s, NUMERICAL_FAILURE, μ, μs, pobj, dobj, pres, dres)
                else
                    #
                    # find the largest step sizes such that
                    #
                    #   p + αp Δp ∈ K
                    #   d + αd Δd ∈ K*
                    #
                    step = @timeit s.timers "maxsteps" maxsteps(s, w.Δp, w.Δd, s.settings.step_frac)
                    #
                    # compute the updated iterates
                    #
                    #   p ← p + α Δp ∈ K
                    #   d ← d + α Δd ∈ K*
                    #
                    axpy!(step, w.Δp, s.p)
                    axpy!(step, w.Δd, s.d)
                    axpy!(step, w.Δy, s.y)
                end
            end
        end
    end

    push!(s.hist, (; μ, step, pres, dres, pobj, dobj, ρ, δ=s.δ[], piter, ppass, pstat, citer, cpass, cstat,
        dmin, dmax, χ))

    return status
end

############################################################################################
# solve / reinit!
############################################################################################

"""
    solve(problem::IPMProblem;
        verbose=false,
        step_frac=0.99,
        feas_tol=1e-8,
        gap_tol=1e-8,
        max_iter=100,
        near_factor=1000.0,
        stall_tol=1e-3,
        refine_stall_tol=0.5,
        scale_max_iter=10,
        refine_max_iter=10,
        newton_max_iter=100,
        aug_tol=1e-7,
    )

Solve an [`IPMProblem`](@ref).
"""
function CommonSolve.solve(prob::IPMProblem{T}, settings::IPMSettings{T}; p0=nothing, d0=nothing, y0=nothing) where {T}
    return solve!(init(prob, settings; p0, d0, y0))
end

function CommonSolve.solve(prob::IPMProblem{T}; kw...) where {T}
    return solve!(init(prob; kw...))
end

