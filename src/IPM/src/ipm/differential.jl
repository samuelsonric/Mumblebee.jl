############################################################################################
# frule!
############################################################################################

function frule!(Δp, Δy, Δd, s::IPMSolver, Δf, Δg, Δμ)
    δ = getaug(s)
    initok, _ = initkkt!(s, δ)
    initok || error()
    μ0 = first(s.hist.μ)                                    # 0 when ν = 0 (no cones anywhere)
    tol = iszero(μ0) ? FORCING_FRAC * s.settings.gap_tol : FORCING_FRAC * last(s.hist.μ) / μ0
    ftol = tol * (1 + s.nf[])
    gtol = tol * (1 + s.ng[])
    return frule!(Δp, Δy, Δd, s.B, s.p, s.y, s.K, s.caches, s.sched,
                  s.P1, s.P2, s.H, s.Q, s.kkt, s.scaling, s.settings, ftol, gtol, Δf, Δg, Δμ)
end

function frule!(
        Δp::AbstractVector,
        Δy::AbstractVector,
        Δd::AbstractVector,
        B::BlockSparseMatrix{T},
        p::FVector{T},
        y::FVector{T},
        K::FVector,
        caches::Caches,
        sched::ConeSchedule,
        P1::FPermutation,
        P2::FPermutation,
        H::BlockSparseMatrix{T},
        Q::BlockSparseMatrix{T},
        kkt::KKTSolver{T},
        scaling::IPMScaling{T},
        settings::IPMSettings{T},
        ftol::T,
        gtol::T,
        Δf::AbstractVector,
        Δg::AbstractVector,
        Δμ,
    ) where {T}
    n = length(p)
    m = length(y)

    ps = scaling.pscl
    ys = scaling.yscl

    Δfx = FVector{T}(undef, n)   # Δf internal;                then reused as f-side RHS Δfx + Δμ d*
    Δgx = FVector{T}(undef, m)   # Δg internal (g-side RHS)
    Δpx = FVector{T}(undef, n)   # Δp internal (solve primal)
    Δyx = FVector{T}(undef, m)   # Δy internal (solve dual)
    Δdx = FVector{T}(undef, n)   # d*(p), then Δμ d*, then Δd internal = Δμ d* − (H−Q) Δpx

    # push the seeds into the internal frame
    mul!(Δfx, P2, Δf)
    Δfx .*= ps

    mul!(Δgx, P1, Δg)
    Δgx .*= ys

    # d* into Δdx;  Δdx ← Δμ d*;  then f-side RHS Δfx += Δμ d*
    dualshadow!(Δdx, B, p, K, caches, sched)
    rmul!(Δdx, Δμ)
    axpy!(1, Δdx, Δfx)

    solvekkt!(kkt, Δpx, Δyx, H, B, Δfx, Δgx;
        warm=false, ftol, gtol, stall=settings.refine_stall_tol,
        irmax=settings.refine_max_iter, cgmax=settings.newton_max_iter, irmin=1)

    # Δdx = Δμ d* − (H − Q) Δpx
    mul!(Δdx, H, Δpx, -1, 1)
    mul!(Δdx, Q, Δpx, 1, 1)

    # pull back to user coordinates
    Δpx .*= ps
    ldiv!(Δp, P2, Δpx)

    Δyx .*= ys
    ldiv!(Δy, P1, Δyx)

    Δdx ./= ps
    ldiv!(Δd, P2, Δdx)

    return Δp, Δy, Δd
end

############################################################################################
# frule2!
############################################################################################

function frule2!(
        Δp1, Δy1, Δd1, Δp2, Δy2, Δd2, Δp12, Δy12, Δd12, s::IPMSolver,
        Δf1, Δg1, Δμ1, Δf2, Δg2, Δμ2,
    )
    δ = getaug(s)
    initok, _ = initkkt!(s, δ)
    initok || error()
    μ0 = first(s.hist.μ)                                    # 0 when ν = 0 (no cones anywhere)
    tol = iszero(μ0) ? FORCING_FRAC * s.settings.gap_tol : FORCING_FRAC * last(s.hist.μ) / μ0
    ftol = tol * (1 + s.nf[])
    gtol = tol * (1 + s.ng[])
    return frule2!(Δp1, Δy1, Δd1, Δp2, Δy2, Δd2, Δp12, Δy12, Δd12,
                   s.B, s.p, s.y, s.K, s.caches, s.sched, s.P1, s.P2, s.H, s.Q, s.kkt,
                   s.scaling, s.settings, s.μ, ftol, gtol, Δf1, Δg1, Δμ1, Δf2, Δg2, Δμ2)
end

function frule2!(
        Δp1::AbstractVector,
        Δy1::AbstractVector,
        Δd1::AbstractVector,
        Δp2::AbstractVector,
        Δy2::AbstractVector,
        Δd2::AbstractVector,
        Δp12::AbstractVector,
        Δy12::AbstractVector,
        Δd12::AbstractVector,
        B::BlockSparseMatrix{T},
        p::FVector{T},
        y::FVector{T},
        K::FVector,
        caches::Caches,
        sched::ConeSchedule,
        P1::FPermutation,
        P2::FPermutation,
        H::BlockSparseMatrix{T},
        Q::BlockSparseMatrix{T},
        kkt::KKTSolver{T},
        scaling::IPMScaling{T},
        settings::IPMSettings{T},
        μ::T,
        ftol::T,
        gtol::T,
        Δf1::AbstractVector,
        Δg1::AbstractVector,
        Δμ1,
        Δf2::AbstractVector,
        Δg2::AbstractVector,
        Δμ2,
    ) where {T}
    n = length(p)
    m = length(y)

    ps = scaling.pscl
    ys = scaling.yscl

    ds    = FVector{T}(undef, n)   # dual shadow d*(p);  then (dead) F″ term scratch
    Δfx   = FVector{T}(undef, n)   # f-side RHS (both seed solves);  then 3rd-order RHS;  then Δd12 internal
    Δgx   = FVector{T}(undef, m)   # g-side RHS (both seed solves);  then zeroed for the 2nd-order solve
    Δp1x  = FVector{T}(undef, n)   # set 1: Δp internal
    Δy1x  = FVector{T}(undef, m)   # set 1: Δy internal
    Δd1x  = FVector{T}(undef, n)   # set 1: Δd internal
    Δp2x  = FVector{T}(undef, n)   # set 2: Δp internal;  then (freed) 2nd-order Δp internal
    Δy2x  = FVector{T}(undef, m)   # set 2: Δy internal;  then (freed) 2nd-order Δy internal
    Δd2x  = FVector{T}(undef, n)   # set 2: Δd internal

    dualshadow!(ds, B, p, K, caches, sched)

    # forward solve, seed set 1  (RHS in Δfx, Δgx)
    mul!(Δfx, P2, Δf1)
    Δfx .*= ps
    axpy!(Δμ1, ds, Δfx)                                     # f-side RHS

    mul!(Δgx, P1, Δg1)
    Δgx .*= ys                                              # g-side RHS

    solvekkt!(kkt, Δp1x, Δy1x, H, B, Δfx, Δgx;
        warm=false, ftol, gtol, stall=settings.refine_stall_tol,
        irmax=settings.refine_max_iter, cgmax=settings.newton_max_iter, irmin=1)

    mul!(Δd1x, H, Δp1x)                                    # Δd1 = Δμ1 d* − (H−Q) Δp1x
    mul!(Δd1x, Q, Δp1x, -1, 1)
    axpby!(Δμ1, ds, -1, Δd1x)

    if (Δf1 === Δf2) && (Δg1 === Δg2) && (Δμ1 === Δμ2)
        copyto!(Δp2x, Δp1x)
        copyto!(Δy2x, Δy1x)
        copyto!(Δd2x, Δd1x)
    else
        # forward solve, seed set 2  (RHS reuses Δfx, Δgx)
        mul!(Δfx, P2, Δf2)
        Δfx .*= ps

        axpy!(Δμ2, ds, Δfx)                                # f-side RHS

        mul!(Δgx, P1, Δg2)
        Δgx .*= ys                                          # g-side RHS

        solvekkt!(kkt, Δp2x, Δy2x, H, B, Δfx, Δgx;
            warm=false, ftol, gtol, stall=settings.refine_stall_tol,
            irmax=settings.refine_max_iter, cgmax=settings.newton_max_iter, irmin=1)

        mul!(Δd2x, H, Δp2x)                               # Δd2 = Δμ2 d* − (H−Q) Δp2x
        mul!(Δd2x, Q, Δp2x, -1, 1)
        axpby!(Δμ2, ds, -1, Δd2x)
    end

    # third-order RHS  Δfx = −( μ′ ∇³F(p)[Δp1, Δp2] + Δμ1 F″(p) Δp2 + Δμ2 F″(p) Δp1 )
    primalthird!(Δfx, B, p, K, caches, sched, Δp1x, Δp2x)
    rmul!(Δfx, μ)

    if !iszero(Δμ1)
        primalhess!(ds, B, p, K, caches, sched, Δp2x)      # ds dead — reuse as F″ scratch
        axpy!(Δμ1, ds, Δfx)
    end

    if !iszero(Δμ2)
        primalhess!(ds, B, p, K, caches, sched, Δp1x)
        axpy!(Δμ2, ds, Δfx)
    end

    rmul!(Δfx, -1)

    # pull back sets 1 and 2  (frees their buffers for the 2nd-order solve)
    Δp1x .*= ps
    ldiv!(Δp1, P2, Δp1x)

    Δy1x .*= ys
    ldiv!(Δy1, P1, Δy1x)

    Δd1x ./= ps
    ldiv!(Δd1, P2, Δd1x)

    Δp2x .*= ps
    ldiv!(Δp2, P2, Δp2x)

    Δy2x .*= ys
    ldiv!(Δy2, P1, Δy2x)

    Δd2x ./= ps
    ldiv!(Δd2, P2, Δd2x)

    # second-order solve  (reuse freed Δp2x, Δy2x;  zero g-side RHS: reuse Δgx)
    fill!(Δgx, false)

    solvekkt!(kkt, Δp2x, Δy2x, H, B, Δfx, Δgx;
        warm=false, ftol, gtol, stall=settings.refine_stall_tol,
        irmax=settings.refine_max_iter, cgmax=settings.newton_max_iter, irmin=1)

    # Δd12 accumulates into Δfx:  Δfx ← Δfx − (H − Q) Δp2x
    mul!(Δfx, H, Δp2x, -1, 1)
    mul!(Δfx, Q, Δp2x, 1, 1)

    # pull back the 2nd-order triple
    Δp2x .*= ps
    ldiv!(Δp12, P2, Δp2x)

    Δy2x .*= ys
    ldiv!(Δy12, P1, Δy2x)

    Δfx ./= ps
    ldiv!(Δd12, P2, Δfx)

    return Δp1, Δy1, Δd1, Δp2, Δy2, Δd2, Δp12, Δy12, Δd12
end

############################################################################################
# rrule!
############################################################################################

function rrule!(Δf, Δg, s::IPMSolver, Δp, Δy, Δd)
    δ = getaug(s)
    initok, _ = initkkt!(s, δ)
    initok || error()
    μ0 = first(s.hist.μ)                                    # 0 when ν = 0 (no cones anywhere)
    tol = iszero(μ0) ? FORCING_FRAC * s.settings.gap_tol : FORCING_FRAC * last(s.hist.μ) / μ0
    ftol = tol * (1 + s.nf[])
    gtol = tol * (1 + s.ng[])
    return rrule!(Δf, Δg, s.B, s.p, s.y, s.K, s.caches, s.sched,
                  s.P1, s.P2, s.H, s.Q, s.kkt, s.scaling, s.settings, ftol, gtol, Δp, Δy, Δd)
end

function rrule!(
        Δf::AbstractVector,
        Δg::AbstractVector,
        B::BlockSparseMatrix{T},
        p::FVector{T},
        y::FVector{T},
        K::FVector,
        caches::Caches,
        sched::ConeSchedule,
        P1::FPermutation,
        P2::FPermutation,
        H::BlockSparseMatrix{T},
        Q::BlockSparseMatrix{T},
        kkt::KKTSolver{T},
        scaling::IPMScaling{T},
        settings::IPMSettings{T},
        ftol::T,
        gtol::T,
        Δp::AbstractVector,
        Δy::AbstractVector,
        Δd::AbstractVector,
    ) where {T}
    n = length(p)
    m = length(y)

    ps = scaling.pscl
    ys = scaling.yscl

    Δpx = FVector{T}(undef, n)   # Δp internal;                then reused as f-side RHS Δpx − (H−Q)Δdx
    Δdx = FVector{T}(undef, n)   # Δd internal;                then reused for λd = Δdx + λ_p
    Δyx = FVector{T}(undef, m)   # −Δy internal (g-side RHS)
    Δfx = FVector{T}(undef, n)   # adjoint primal λ_p;         then scaled to Δf internal
    Δgx = FVector{T}(undef, m)   # adjoint dual;               then scaled to Δg internal (λ_y = −ξ)

    # push the seeds into the internal frame
    mul!(Δpx, P2, Δp)
    Δpx .*= ps

    mul!(Δdx, P2, Δd)
    Δdx ./= ps

    mul!(Δyx, P1, Δy)
    Δyx .*= -ys                                            # −Δy internal

    # Δpx −= (H − Q) Δdx
    mul!(Δpx, H, Δdx, -1, 1)
    mul!(Δpx, Q, Δdx, 1, 1)

    solvekkt!(kkt, Δfx, Δgx, H, B, Δpx, Δyx;
        warm=false, ftol, gtol, stall=settings.refine_stall_tol,
        irmax=settings.refine_max_iter, cgmax=settings.newton_max_iter, irmin=1)

    axpy!(1, Δfx, Δdx)                                    # Δdx = λd = Δd internal + λ_p

    Δfx .*= ps                                            # Δf internal = ps λ_p
    ldiv!(Δf, P2, Δfx)

    Δgx .*= -ys                                           # Δg internal = −ys ξ
    ldiv!(Δg, P1, Δgx)

    dualshadow!(Δpx, B, p, K, caches, sched)              # reuse Δpx (free) for d*(p)
    dμ = dot(Δdx, Δpx)

    return Δf, Δg, dμ
end
