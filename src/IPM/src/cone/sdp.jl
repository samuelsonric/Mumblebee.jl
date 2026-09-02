"""
    SemidefiniteCone <: AbstractCone

A cone of n × n positive-semidefinite
matrices.
"""
struct SemidefiniteCone <: AbstractCone end

# Bisection bracket width for the tridiagonal step-length path. eigminsturm!
# certifies a lower bound on λmin (no Cholesky feasibility gate needed), so τ is
# only ever short — never over — by at most this bracket.
const TRIDIAG_TOL = 1e-6

# SDP workspace layout in data (d = triroot(n)):
#   sdpscale!:              2 d²       (P, D; the SVD runs in place in the R cache)
#   sdpcorr!:               3 d²       (ΔP, ΔD, W)
#   sdpmaxstep tridiag:     2 d² + 2 d   (α doubles as the tridiag scratch)
# The bound is max(3 d², 2 d² + 2 d): sdpcorr! wins for d ≥ 2, the tridiag
# maxstep only at d = 1.
function workspacesize(::Type{SemidefiniteCone}, n::Integer)
    d = triroot(n)
    return max(3d^2, 2d^2 + 2d)
end

struct SemidefiniteConeCache{T} <: AbstractCache{SemidefiniteCone}
    cone::SemidefiniteCone
    #
    # R = LP⁻ᵀ U, where P = LP LPᵀ and LPᵀ LD = U Σ Vᵀ.
    # Satisfies R Rᵀ = P⁻¹. The corrector's congruences and the closing map
    # (R · Rᵀ) are all expressed through it; the maxstep τp = eigmin(Rᵀ ΔP R).
    #
    R::FMatrixView{T}
    #
    # S = LP U = R⁻ᵀ. Satisfies S Sᵀ = P and Rᵀ S = I.
    # The dual congruence B = Sᵀ ΔD S and τd = eigmin(Σ⁻¹ B Σ⁻¹) use it.
    #
    S::FMatrixView{T}
    #
    # the singular values Σ of LPᵀ LD
    #
    s::FVectorView{T}
end

function triroot(n::Integer)
    return (isqrt(1 + 8n) - 1) ÷ 2
end

function roottwo(::Type{T}) where {T}
    return sqrt(two(T))
end

# compute the symmetric Kronecker product
#
#   H = W⁻¹ ⊗ W⁻¹
#
# where W is the Nesterov-Todd scaling point
#
#   W = √P √(√P D √P)⁻¹ √P
#     = √D √(√D P √D)⁻¹ √D
#
function sdpscalestatic!(
        H::AbstractMatrix{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        p::AbstractVector,
        d::AbstractVector,
        wrk::ConeWorkspace{T},
        ::Val{N},
    ) where {T, N}
    m = N * N
    P = reshape(view(wrk.data, 0m + 1:1m), N, N)   # chol(smat p); reused for Y after S
    D = reshape(view(wrk.data, 1m + 1:2m), N, N)   # chol(smat d); → W

    @inbounds smatstatic!(P, p, Val(N))
    @inbounds smatstatic!(D, d, Val(N))

    @inbounds cholstatic!(Symmetric(P, :L), Val(N)) || return false
    @inbounds cholstatic!(Symmetric(D, :L), Val(N)) || return false
    #
    # svd of Pᵀ D straight into the R cache: R ← Pᵀ D, then
    # svd overwrites R with its left singular vectors U.
    # From there S = P U, then R = P⁻ᵀ U in place.
    #
    @inbounds copystatic!(R, LowerTriangular(D), Val(N))
    @inbounds lmulstatic!(LowerTriangular(P)', R, Val(N))     # R = Pᵀ D
    @inbounds svdjacobi!(R, s) || return false

    @inbounds copystatic!(S, R, Val(N))                       # S = U
    @inbounds lmulstatic!(LowerTriangular(P), S, Val(N))      # S = P U
    @inbounds ldivstatic!(LowerTriangular(P)', R, Val(N))     # R = P⁻ᵀ U
    #
    # W = R Σ Rᵀ = (R Σ^½)(R Σ^½)ᵀ:  Y = R Σ^½ into P, W into D
    #
    @inbounds for j in 1:N
        α = sqrt(s[j])

        for i in 1:N
            P[i, j] = R[i, j] * α
        end
    end

    @inbounds syrkstatic!(D, P, Val(N))
    @inbounds symmstatic!(D, Val(N))
    @inbounds skronstatic!(H, D, Val(N))

    return true
end

function sdpscaledynamic!(
        H::AbstractMatrix{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        p::AbstractVector,
        d::AbstractVector,
        wrk::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1); m = n * n

    P = reshape(view(wrk.data, 0m + 1:1m), n, n)   # chol(smat p); reused for Y after S
    D = reshape(view(wrk.data, 1m + 1:2m), n, n)   # chol(smat d); → W

    smat!(P, p)
    smat!(D, d)

    FP = cholesky!(Symmetric(P, :L); check=false)
    issuccess(FP) || return false

    FD = cholesky!(Symmetric(D, :L); check=false)
    issuccess(FD) || return false

    # svd of Pᵀ D straight into the R cache: R = left singular vectors U,
    # then S = P U and R = P⁻ᵀ U in place.
    copyto!(R, LowerTriangular(D))
    lmul!(LowerTriangular(P)', R)     # R = Pᵀ D
    svdjacobi!(R, s)                  # R = left singular vectors

    copyto!(S, R)
    lmul!(LowerTriangular(P), S)      # S = P U
    ldiv!(LowerTriangular(P)', R)     # R = P⁻ᵀ U

    # W = R Σ Rᵀ:  Y = R Σ^½ into P, W into D
    @inbounds for j in 1:n
        α = sqrt(s[j])

        for i in 1:n
            P[i, j] = R[i, j] * α
        end
    end

    syrk!(D, P)
    symmetrize!(D)
    skron!(H, D)

    return true
end

function sdpscale!(
        H::AbstractMatrix{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        p::AbstractVector,
        d::AbstractVector,
        wrk::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1)
    n == 1 && return sdpscalestatic!(H, R, S, s, p, d, wrk, Val(1))
    n == 2 && return sdpscalestatic!(H, R, S, s, p, d, wrk, Val(2))
    n == 3 && return sdpscalestatic!(H, R, S, s, p, d, wrk, Val(3))
    n == 4 && return sdpscalestatic!(H, R, S, s, p, d, wrk, Val(4))
    n == 5 && return sdpscalestatic!(H, R, S, s, p, d, wrk, Val(5))
    n == 6 && return sdpscalestatic!(H, R, S, s, p, d, wrk, Val(6))
    n == 7 && return sdpscalestatic!(H, R, S, s, p, d, wrk, Val(7))
    n == 8 && return sdpscalestatic!(H, R, S, s, p, d, wrk, Val(8))
    return sdpscaledynamic!(H, R, S, s, p, d, wrk)
end

# The SDP Mehrotra corrector. With R = LP⁻ᵀU, S = LP U (= R⁻ᵀ) and Σ = s cached,
#
#   r = svec( R (σμ Σ⁻¹ − Σ − 𝓛⁻¹(X)) Rᵀ )
#
# where the coupling matrix factors into two congruences,
#
#   X = A B,   A = Rᵀ ΔP R,   B = Sᵀ ΔD S,
#
# and 𝓛 is the Lyapunov operator 𝓛(Y) = ΣY + YΣ. No triangular solves and no
# square roots of Σ are formed (the √Σ split of the NT factor fuses into the
# weightedmean step below). n ≤ 8 unrolls via the static kernels, else BLAS.
function sdpcorr!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        wrk::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1)

    n == 1 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, wrk, Val(1))
    n == 2 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, wrk, Val(2))
    n == 3 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, wrk, Val(3))
    n == 4 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, wrk, Val(4))
    n == 5 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, wrk, Val(5))
    n == 6 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, wrk, Val(6))
    n == 7 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, wrk, Val(7))
    n == 8 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, wrk, Val(8))
    return sdpcorrdynamic!(r, R, S, s, Δp, Δd, σμ, wrk)
end

function sdpcorrstatic!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        wrk::ConeWorkspace{T},
        ::Val{N},
    ) where {T, N}
    m = N * N

    ΔP = reshape(view(wrk.data, 0m + 1:1m), N, N)
    ΔD = reshape(view(wrk.data, 1m + 1:2m), N, N)
    W  = reshape(view(wrk.data, 2m + 1:3m), N, N)   # intermediate, then A·B, then Ŵ, then R Ŵ Rᵀ

    @inbounds smatstatic!(ΔP, Δp, Val(N))
    @inbounds smatstatic!(ΔD, Δd, Val(N))
    @inbounds symmstatic!(ΔP, Val(N))
    @inbounds symmstatic!(ΔD, Val(N))
    #
    #   A = Rᵀ ΔP R   (ΔP overwritten with A; W the intermediate)
    #
    @inbounds mulstatic!(W, ΔP, R, Val(N))
    @inbounds mulstatic!(ΔP, R', W, Val(N))
    #
    #   B = Sᵀ ΔD S   (ΔD overwritten with B)
    #
    @inbounds mulstatic!(W, ΔD, S, Val(N))
    @inbounds mulstatic!(ΔD, S', W, Val(N))
    #
    #   W = A B   (not symmetric — both W[i,j] and W[j,i] are read below)
    #
    @inbounds mulstatic!(W, ΔP, ΔD, Val(N))
    #
    # Ŵ = Σ^½ (σμ Σ⁻¹ − Σ − 𝓛⁻¹(W)) Σ^½, overwriting W in place — each pair is
    # read then written (see sdpcorrdynamic! for the derivation)
    #
    @inbounds for j in 1:N
        sj = s[j]

        for i in 1:j - 1
            W[i, j] = W[j, i] = -weightedmean(s[i], sj, W[i, j], W[j, i])
        end

        W[j, j] = σμ - sj^2 - W[j, j]
    end
    #
    #   W = R Ŵ Rᵀ,  r = svec(W)   (ΔP dead, reused as the intermediate)
    #
    @inbounds mulstatic!(ΔP, W, R', Val(N))
    @inbounds mulstatic!(W, R, ΔP, Val(N))

    @inbounds svecstatic!(r, W, Val(N))
    return r
end

function sdpcorrdynamic!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        wrk::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1); m = n * n

    ΔP = reshape(view(wrk.data, 0m + 1:1m), n, n)
    ΔD = reshape(view(wrk.data, 1m + 1:2m), n, n)
    W  = reshape(view(wrk.data, 2m + 1:3m), n, n)   # intermediate, then A·B, then Ŵ, then R Ŵ Rᵀ

    smat!(ΔP, Δp)
    smat!(ΔD, Δd)
    symmetrize!(ΔP)
    symmetrize!(ΔD)
    #
    #   A = Rᵀ ΔP R   (ΔP overwritten with A; W the intermediate)
    #
    mul!(W, ΔP, R)
    mul!(ΔP, R', W)
    #
    #   B = Sᵀ ΔD S   (ΔD overwritten with B)
    #
    mul!(W, ΔD, S)
    mul!(ΔD, S', W)
    #
    #   W = A B   (not symmetric — both W[i,j] and W[j,i] are read below)
    #
    mul!(W, ΔP, ΔD)
    #
    # Ŵ = Σ^½ (σμ Σ⁻¹ − Σ − 𝓛⁻¹(W)) Σ^½, formed in place over W: 𝓛⁻¹ divides the
    # off-diagonal by (sᵢ+sⱼ) and the √Σ factors of R = P⁻ᵀU (= R̃ Σ^½ in NT
    # form) fuse in here via weightedmean, so no square roots of Σ are taken.
    # Each pair is read then written.
    #
    for j in 1:n
        sj = s[j]

        for i in 1:j - 1
            W[i, j] = W[j, i] = -weightedmean(s[i], sj, W[i, j], W[j, i])
        end

        W[j, j] = σμ - sj^2 - W[j, j]
    end
    #
    #   W = R Ŵ Rᵀ,  r = svec(W)   (ΔP dead, reused as the intermediate)
    #
    mul!(ΔP, W, R')
    mul!(W, R, ΔP)

    svec!(r, W)
    return r
end

# Find the largest number 0 < τ ≤ 1 such that
#
#   L Lᵀ + τ ΔX = L (I + τ M) Lᵀ
#
# is positive definite, where
#
#   M = L⁻¹ ΔX L⁻ᵀ.
#
# This matrix is positive definite if and
# only if M is, so the solution is given by
#
#   τ⁻¹ = max {1, -λ},
#
# where λ is the smallest eigenvalue of L⁻¹ ΔX L⁻ᵀ.
# construct the identity matrix
#
#   I
#
function sdpid!(x::AbstractVector{T}) where {T}
    d = triroot(length(x))
    k = 1

    fill!(x, zero(T))

    for j in 1:d
        x[k] = one(T); k += d - j + 1
    end

    return x
end

#
# Tiered step-length computation: dispatch by matrix size. The step matrix M is
# an orthogonal congruence of the factor F (F = R for τp, F = S for τd), plus a
# Σ⁻¹·Σ⁻¹ scaling for the dual. M has the same spectrum as LP⁻¹ΔP LP⁻ᵀ /
# LD⁻¹ΔD LD⁻ᵀ, so the step length is unchanged — no triangular factor is needed.
#
# n ≤ 8:            unrolled congruence; closed-form eigmin (n ≤ 3) or
#                   Householder tridiagonalization + Sturm bisection (n ≥ 4)
# n ≥ 9:            dynamic congruence + tridiagonalization + Sturm
#

function sdpmaxstepstatic(F::AbstractMatrix{T}, Δx::AbstractVector{T}, scale, wrk::ConeWorkspace{T}, ::Val{N}) where {T, N}
    m = N * N
    M   = reshape(view(wrk.data, 0m + 1:1m), N, N)
    tmp = reshape(view(wrk.data, 1m + 1:2m), N, N)

    @inbounds smatstatic!(M, Δx, Val(N))
    @inbounds symmstatic!(M, Val(N))
    @inbounds mulstatic!(tmp, M, F, Val(N))          # tmp = ΔX F
    @inbounds mulstatic!(M, F', tmp, Val(N))         # M = Fᵀ ΔX F

    if !isnothing(scale)
        @inbounds for j in 1:N, i in 1:N
            M[i, j] /= scale[i] * scale[j]
        end
    end
    #
    # closed-form eigmin for n ≤ 3, else Householder tridiag + Sturm bisection
    # (α, β carved from the workspace tail past M and tmp; α doubles as the
    # tridiagonalization scratch)
    #
    if N ≤ 3
        @inbounds λ = eigminstatic(M, Val(N))
    else
        α = view(wrk.data, 2m + 0N + 1:2m + 1N)
        β = view(wrk.data, 2m + 1N + 1:2m + 2N)
        @inbounds λ = eigminsturm!(M, α, β, T(TRIDIAG_TOL), -one(T))
    end

    return inv(max(one(T), -λ))
end

# n ≥ 9: Householder tridiagonalization (exact orthogonal similarity) + Sturm
# bisection. Sturm counting makes `lo` a certified lower bound on λmin, so —
# unlike the old Kato-Temple/Lanczos path — no Cholesky feasibility gate exists.
# Beats LAPACK dsyevr at every size measured (to n=64+), so there is no dense tier.
function sdpmaxstepdynamic(F::AbstractMatrix{T}, Δx::AbstractVector{T}, scale, wrk::ConeWorkspace{T}; tol::T = T(TRIDIAG_TOL)) where {T}
    n = size(F, 1)

    o = 0
    M   = reshape(view(wrk.data, o + 1:o + n * n), n, n); o += n * n
    tmp = reshape(view(wrk.data, o + 1:o + n * n), n, n); o += n * n
    α =         view(wrk.data, o + 1:o + n);              o += n
    β =         view(wrk.data, o + 1:o + n)
    #
    # M = Fᵀ ΔX F (+ Σ⁻¹·Σ⁻¹ for the dual): ΔX built in M, congruence via tmp
    #
    smat!(M, Δx)
    symmetrize!(M)
    mul!(tmp, M, F)
    mul!(M, F', tmp)

    if !isnothing(scale)
        @inbounds for j in 1:n, i in 1:n
            M[i, j] /= scale[i] * scale[j]
        end
    end

    @inbounds λ = eigminsturm!(M, α, β, tol, -one(T))

    return inv(max(one(T), -λ))
end

# Dispatcher
function sdpmaxstep(F::AbstractMatrix{T}, Δx::AbstractVector{T}, scale, wrk::ConeWorkspace{T}) where {T}
    n = size(F, 1)

    n == 1 && return sdpmaxstepstatic(F, Δx, scale, wrk, Val(1))
    n == 2 && return sdpmaxstepstatic(F, Δx, scale, wrk, Val(2))
    n == 3 && return sdpmaxstepstatic(F, Δx, scale, wrk, Val(3))
    n == 4 && return sdpmaxstepstatic(F, Δx, scale, wrk, Val(4))
    n == 5 && return sdpmaxstepstatic(F, Δx, scale, wrk, Val(5))
    n == 6 && return sdpmaxstepstatic(F, Δx, scale, wrk, Val(6))
    n == 7 && return sdpmaxstepstatic(F, Δx, scale, wrk, Val(7))
    n == 8 && return sdpmaxstepstatic(F, Δx, scale, wrk, Val(8))
    return sdpmaxstepdynamic(F, Δx, scale, wrk)
end

#
# AbstractCone Interface
#

function degree(::SemidefiniteCone, n::Integer)
    return triroot(n)
end

function cachesize(::Type{SemidefiniteCone}, n::Integer)
    d = triroot(n)
    return 2d^2 + d  # R, S (d² each) + s (d)
end

function cache(c::Caches, i::Integer, cone::SemidefiniteCone)
    n = c.xcol[i + 1] - c.xcol[i]
    d = triroot(n)

    data = cachedata(c, i)

    R  = reshape(view(data, 0d^2 + 1:1d^2      ), d, d)
    S  = reshape(view(data, 1d^2 + 1:2d^2      ), d, d)
    s  =         view(data, 2d^2 + 1:2d^2 +  d)

    SemidefiniteConeCache(cone, R, S, s)
end

function identity!(x::AbstractVector, ::SemidefiniteCone)
    return sdpid!(x)
end

function scale!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}, cache::SemidefiniteConeCache{T}, wrk::ConeWorkspace{T}) where {T}
    return sdpscale!(H, cache.R, cache.S, cache.s, p, d, wrk)
end

function corr!(
        r::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        cache::SemidefiniteConeCache{T},
        wrk::ConeWorkspace{T},
    ) where {T}
    return sdpcorr!(r, cache.R, cache.S, cache.s, Δp, Δd, σμ, wrk)
end

function maxsteps(::AbstractVector{T}, Δp::AbstractVector{T}, ::AbstractVector{T}, Δd::AbstractVector{T}, cache::SemidefiniteConeCache{T}, wrk::ConeWorkspace{T}) where {T}
    τp = sdpmaxstep(cache.R, Δp, nothing, wrk)
    τd = sdpmaxstep(cache.S, Δd, cache.s, wrk)
    return τp, τd
end

function sdpshadow!(s::AbstractVector{T}, x::AbstractVector{T}, wrk::ConeWorkspace{T}) where {T}
    d = triroot(length(x))

    M = reshape(view(wrk.data, 0d^2 + 1:1d^2), d, d)
    Z = reshape(view(wrk.data, 1d^2 + 1:2d^2), d, d)

    d == 1 && return sdpshadowstatic!(s, x, M, Z, Val(1))
    d == 2 && return sdpshadowstatic!(s, x, M, Z, Val(2))
    d == 3 && return sdpshadowstatic!(s, x, M, Z, Val(3))
    d == 4 && return sdpshadowstatic!(s, x, M, Z, Val(4))
    d == 5 && return sdpshadowstatic!(s, x, M, Z, Val(5))
    d == 6 && return sdpshadowstatic!(s, x, M, Z, Val(6))
    d == 7 && return sdpshadowstatic!(s, x, M, Z, Val(7))
    d == 8 && return sdpshadowstatic!(s, x, M, Z, Val(8))

    return sdpshadowdynamic!(s, x, M, Z)
end

function sdpshadowstatic!(s::AbstractVector{T}, x::AbstractVector{T}, M::AbstractMatrix{T}, Z::AbstractMatrix{T}, ::Val{N}) where {T, N}
    @inbounds smatstatic!(M, x, Val(N))
    @inbounds cholstatic!(Symmetric(M, :L), Val(N)) || return false

    @inbounds for j in 1:N, i in 1:N
        Z[i, j] = ifelse(i == j, one(T), zero(T))
    end

    @inbounds ldivstatic!(LowerTriangular(M), Z, Val(N))
    @inbounds ldivstatic!(LowerTriangular(M)', Z, Val(N))
    @inbounds svecstatic!(s, Z, Val(N))
    return true
end

function sdpshadowdynamic!(s::AbstractVector{T}, x::AbstractVector{T}, M::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T}
    @inbounds smat!(M, x)

    F = cholesky!(Symmetric(M, :L); check=false)
    issuccess(F) || return false

    copyto!(Z, I)
    ldiv!(LowerTriangular(M), Z)
    ldiv!(LowerTriangular(M)', Z)

    @inbounds svec!(s, Z)
    return true
end

function dualshadow!(sd::AbstractVector, p::AbstractVector, ::SemidefiniteConeCache, wrk::ConeWorkspace)
    return sdpshadow!(sd, p, wrk)
end

function primalshadow!(sp::AbstractVector, d::AbstractVector, ::SemidefiniteConeCache, wrk::ConeWorkspace)
    return sdpshadow!(sp, d, wrk)
end
