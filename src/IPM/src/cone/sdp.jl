"""
    SemidefiniteCone <: AbstractCone

A cone of n × n positive-semidefinite
matrices.
"""
struct SemidefiniteCone <: AbstractCone end


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

############################################################################################
# degree
############################################################################################

function degree(::SemidefiniteCone, n::Integer)
    return triroot(n)
end

############################################################################################
# cachesize
############################################################################################

function cachesize(::Type{SemidefiniteCone}, n::Integer)
    d = triroot(n)
    return 2d^2 + d  # R, S (d² each) + s (d)
end

############################################################################################
# workspacesize
############################################################################################

# SDP workspace layout in data (d = triroot(n)):
#   sdpscale!:              2 d²       (P, D; the SVD runs in place in the R cache)
#   sdpcorr!:               3 d²       (ΔP, ΔD, W)
#   sdpmaxstep tridiag:     2 d² + 2 d   (α doubles as the tridiag scratch; C
#                                          doubles as the τ=1 certificate scratch,
#                                          its ΔX·F liveness ending at the congruence)
# The bound is max(3 d², 2 d² + 2 d): sdpcorr! wins for d ≥ 2, the tridiag
# maxstep only at d = 1.
function workspacesize(::Type{SemidefiniteCone}, n::Integer)
    d = triroot(n)
    return max(3d^2, 2d^2 + 2d)
end

############################################################################################
# cache
############################################################################################

function cache(c::Caches, i::Integer, cone::SemidefiniteCone)
    n = c.xcol[i + 1] - c.xcol[i]
    d = triroot(n)

    data = cachedata(c, i)

    R  = reshape(view(data, 0d^2 + 1:1d^2      ), d, d)
    S  = reshape(view(data, 1d^2 + 1:2d^2      ), d, d)
    s  =         view(data, 2d^2 + 1:2d^2 +  d)

    SemidefiniteConeCache(cone, R, S, s)
end

############################################################################################
# identity!
############################################################################################

function identity!(x::AbstractVector, ::SemidefiniteCone)
    return sdpid!(x)
end

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

############################################################################################
# scale!
############################################################################################

function scale!(H::AbstractMatrix{T}, p::AbstractVector{T}, d::AbstractVector{T}, cache::SemidefiniteConeCache{T}, work::ConeWorkspace{T}) where {T}
    return sdpscale!(H, cache.R, cache.S, cache.s, p, d, work)
end

function sdpscale!(
        H::AbstractMatrix{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        p::AbstractVector,
        d::AbstractVector,
        work::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1)
    n == 1 && return sdpscalestatic!(H, R, S, s, p, d, work, Val(1))
    n == 2 && return sdpscalestatic!(H, R, S, s, p, d, work, Val(2))
    n == 3 && return sdpscalestatic!(H, R, S, s, p, d, work, Val(3))
    n == 4 && return sdpscalestatic!(H, R, S, s, p, d, work, Val(4))
    n == 5 && return sdpscalestatic!(H, R, S, s, p, d, work, Val(5))
    n == 6 && return sdpscalestatic!(H, R, S, s, p, d, work, Val(6))
    n == 7 && return sdpscalestatic!(H, R, S, s, p, d, work, Val(7))
    n == 8 && return sdpscalestatic!(H, R, S, s, p, d, work, Val(8))
    return sdpscaledynamic!(H, R, S, s, p, d, work)
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
        work::ConeWorkspace{T},
        n::Val{N},
    ) where {T, N}
    m = N * N
    P = reshape(view(work.data, 0m + 1:1m), N, N)   # chol(smat p); reused for Y after S
    D = reshape(view(work.data, 1m + 1:2m), N, N)   # chol(smat d); → W

    @inbounds smatstatic!(P, p, n)
    @inbounds smatstatic!(D, d, n)

    @inbounds cholstatic!(Symmetric(P, :L), n) || return false, zero(T)
    @inbounds cholstatic!(Symmetric(D, :L), n) || return false, zero(T)
    #
    # svd of Pᵀ D straight into the R cache: R ← Pᵀ D, then
    # svd overwrites R with its left singular vectors U.
    # From there S = P U, then R = P⁻ᵀ U in place.
    #
    @inbounds copystatic!(R, LowerTriangular(D), n)
    @inbounds lmulstatic!(LowerTriangular(P)', R, n)     # R = Pᵀ D
    @inbounds svdjacobi!(R, s) || return false, zero(T)

    @inbounds copystatic!(S, R, n)                       # S = U
    @inbounds lmulstatic!(LowerTriangular(P), S, n)      # S = P U
    @inbounds ldivstatic!(LowerTriangular(P)', R, n)     # R = P⁻ᵀ U
    #
    # W = R Σ Rᵀ = (R Σ^½)(R Σ^½)ᵀ:  Y = R Σ^½ into P, W into D
    #
    @inbounds for j in 1:N
        α = sqrt(s[j])

        for i in 1:N
            P[i, j] = R[i, j] * α
        end
    end

    @inbounds syrkstatic!(D, P, n)
    @inbounds symmstatic!(D, n)
    @inbounds skronstatic!(H, D, n)
    #
    # ⟨p*, d*⟩ = tr(P⁻¹ D⁻¹) = Σ 1/sᵢ²  (sᵢ² = μ on the central path)
    #
    spsd = zero(T)

    @inbounds for i in 1:N
        spsd += inv(s[i]^2)
    end

    return true, spsd
end

function sdpscaledynamic!(
        H::AbstractMatrix{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        p::AbstractVector,
        d::AbstractVector,
        work::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1); m = n * n

    P = reshape(view(work.data, 0m + 1:1m), n, n)   # chol(smat p); reused for Y after S
    D = reshape(view(work.data, 1m + 1:2m), n, n)   # chol(smat d); → W

    smat!(P, p)
    smat!(D, d)

    FP = cholesky!(Symmetric(P, :L); check=false)
    issuccess(FP) || return false, zero(T)

    FD = cholesky!(Symmetric(D, :L); check=false)
    issuccess(FD) || return false, zero(T)

    # svd of Pᵀ D straight into the R cache: R = left singular vectors U,
    # then S = P U and R = P⁻ᵀ U in place.
    copyto!(R, LowerTriangular(D))
    lmul!(LowerTriangular(P)', R)     # R = Pᵀ D
    svdjacobi!(R, s) || return false, zero(T)   # R = left singular vectors

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
    #
    # ⟨p*, d*⟩ = tr(P⁻¹ D⁻¹) = Σ 1/sᵢ²  (sᵢ² = μ on the central path)
    #
    spsd = zero(T)

    @inbounds for i in eachindex(s)
        spsd += inv(s[i]^2)
    end

    return true, spsd
end

############################################################################################
# corr!
############################################################################################

function corr!(
        r::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        cache::SemidefiniteConeCache{T},
        work::ConeWorkspace{T},
    ) where {T}
    return sdpcorr!(r, cache.R, cache.S, cache.s, Δp, Δd, σμ, work)
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
        work::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1)

    n == 1 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, work, Val(1))
    n == 2 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, work, Val(2))
    n == 3 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, work, Val(3))
    n == 4 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, work, Val(4))
    n == 5 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, work, Val(5))
    n == 6 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, work, Val(6))
    n == 7 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, work, Val(7))
    n == 8 && return sdpcorrstatic!(r, R, S, s, Δp, Δd, σμ, work, Val(8))
    return sdpcorrdynamic!(r, R, S, s, Δp, Δd, σμ, work)
end

function sdpcorrstatic!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        S::AbstractMatrix{T},
        s::AbstractVector{T},
        Δp::AbstractVector{T},
        Δd::AbstractVector{T},
        σμ::Real,
        work::ConeWorkspace{T},
        n::Val{N},
    ) where {T, N}
    m = N * N

    ΔP = reshape(view(work.data, 0m + 1:1m), N, N)
    ΔD = reshape(view(work.data, 1m + 1:2m), N, N)
    W  = reshape(view(work.data, 2m + 1:3m), N, N)   # intermediate, then A·B, then Ŵ, then R Ŵ Rᵀ

    @inbounds smatstatic!(ΔP, Δp, n)
    @inbounds smatstatic!(ΔD, Δd, n)
    @inbounds symmstatic!(ΔP, n)
    @inbounds symmstatic!(ΔD, n)
    #
    #   A = Rᵀ ΔP R   (ΔP overwritten with A; W the intermediate)
    #
    @inbounds mulstatic!(W, ΔP, R, n)
    @inbounds mulstatic!(ΔP, R', W, n)
    #
    #   B = Sᵀ ΔD S   (ΔD overwritten with B)
    #
    @inbounds mulstatic!(W, ΔD, S, n)
    @inbounds mulstatic!(ΔD, S', W, n)
    #
    #   W = A B   (not symmetric — both W[i,j] and W[j,i] are read below)
    #
    @inbounds mulstatic!(W, ΔP, ΔD, n)
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
    @inbounds mulstatic!(ΔP, W, R', n)
    @inbounds mulstatic!(W, R, ΔP, n)

    @inbounds svecstatic!(r, W, n)
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
        work::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1); m = n * n

    ΔP = reshape(view(work.data, 0m + 1:1m), n, n)
    ΔD = reshape(view(work.data, 1m + 1:2m), n, n)
    W  = reshape(view(work.data, 2m + 1:3m), n, n)   # intermediate, then A·B, then Ŵ, then R Ŵ Rᵀ

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

############################################################################################
# corr0!
############################################################################################

function corr0!(r::AbstractVector{T}, ::AbstractVector{T}, ::AbstractVector{T}, σμ::Real, cache::SemidefiniteConeCache{T}, work::ConeWorkspace{T}) where {T}
    return sdpcorr0!(r, cache.R, cache.s, σμ, work)
end

# sdpcorr! with Δp = Δd = 0.
function sdpcorr0!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        s::AbstractVector{T},
        σμ::Real,
        work::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1)

    n == 1 && return sdpcorr0static!(r, R, s, σμ, work, Val(1))
    n == 2 && return sdpcorr0static!(r, R, s, σμ, work, Val(2))
    n == 3 && return sdpcorr0static!(r, R, s, σμ, work, Val(3))
    n == 4 && return sdpcorr0static!(r, R, s, σμ, work, Val(4))
    n == 5 && return sdpcorr0static!(r, R, s, σμ, work, Val(5))
    n == 6 && return sdpcorr0static!(r, R, s, σμ, work, Val(6))
    n == 7 && return sdpcorr0static!(r, R, s, σμ, work, Val(7))
    n == 8 && return sdpcorr0static!(r, R, s, σμ, work, Val(8))
    return sdpcorr0dynamic!(r, R, s, σμ, work)
end

function sdpcorr0static!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        s::AbstractVector{T},
        σμ::Real,
        work::ConeWorkspace{T},
        n::Val{N},
    ) where {T, N}
    m = N * N

    ΔP = reshape(view(work.data, 0m + 1:1m), N, N)
    W  = reshape(view(work.data, 2m + 1:3m), N, N)

    @inbounds for j in 1:N
        for i in 1:N
            W[i, j] = zero(T)
        end

        W[j, j] = σμ - s[j]^2
    end
    #
    #   W = R Ŵ Rᵀ,  r = svec(W)   (ΔP the intermediate)
    #
    @inbounds mulstatic!(ΔP, W, R', n)
    @inbounds mulstatic!(W, R, ΔP, n)

    @inbounds svecstatic!(r, W, n)
    return r
end

function sdpcorr0dynamic!(
        r::AbstractVector{T},
        R::AbstractMatrix{T},
        s::AbstractVector{T},
        σμ::Real,
        work::ConeWorkspace{T},
    ) where {T}
    n = size(R, 1); m = n * n

    ΔP = reshape(view(work.data, 0m + 1:1m), n, n)
    W  = reshape(view(work.data, 2m + 1:3m), n, n)

    fill!(W, zero(T))

    for j in 1:n
        W[j, j] = σμ - s[j]^2
    end

    mul!(ΔP, W, R')
    mul!(W, R, ΔP)

    svec!(r, W)
    return r
end

############################################################################################
# maxsteps
############################################################################################

function maxsteps(::AbstractVector{T}, Δp::AbstractVector{T}, ::AbstractVector{T}, Δd::AbstractVector{T}, cache::SemidefiniteConeCache{T}, work::ConeWorkspace{T}) where {T}
    τp = sdpmaxstep(cache.R, Δp, nothing, work)
    τd = sdpmaxstep(cache.S, Δd, cache.s, work)
    return τp, τd
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
#
function sdpmaxstep(F::AbstractMatrix{T}, Δx::AbstractVector{T}, scale, work::ConeWorkspace{T}) where {T}
    n = size(F, 1)

    n == 1 && return sdpmaxstepstatic(F, Δx, scale, work, Val(1))
    n == 2 && return sdpmaxstepstatic(F, Δx, scale, work, Val(2))
    n == 3 && return sdpmaxstepstatic(F, Δx, scale, work, Val(3))
    n == 4 && return sdpmaxstepstatic(F, Δx, scale, work, Val(4))
    n == 5 && return sdpmaxstepstatic(F, Δx, scale, work, Val(5))
    n == 6 && return sdpmaxstepstatic(F, Δx, scale, work, Val(6))
    n == 7 && return sdpmaxstepstatic(F, Δx, scale, work, Val(7))
    n == 8 && return sdpmaxstepstatic(F, Δx, scale, work, Val(8))
    return sdpmaxstepdynamic(F, Δx, scale, work)
end

function sdpmaxstepstatic(F::AbstractMatrix{T}, Δx::AbstractVector{T}, scale, work::ConeWorkspace{T}, n::Val{N}) where {T, N}
    m = N * N
    M = reshape(view(work.data, 0m + 1:1m), N, N)
    C = reshape(view(work.data, 1m + 1:2m), N, N)

    @inbounds smatstatic!(M, Δx, n)
    @inbounds symmstatic!(M, n)
    @inbounds mulstatic!(C, M, F, n)          # C = ΔX F
    @inbounds mulstatic!(M, F', C, n)         # M = Fᵀ ΔX F

    if !isnothing(scale)
        @inbounds for j in 1:N, i in 1:N
            M[i, j] /= scale[i] * scale[j]
        end
    end

    @inbounds for j in 1:N
        for i in j:N
            C[i, j] = M[i, j]
        end

        C[j, j] += one(T)
    end

    if @inbounds cholstatic!(Symmetric(C, :L), n)
        return one(T)
    end

    if N ≤ 3
        @inbounds λ = eigminstatic(M, n)
    else
        α = view(work.data, 2m + 0N + 1:2m + 1N)
        β = view(work.data, 2m + 1N + 1:2m + 2N)
        @inbounds λ = eigminsturm!(M, α, β, -one(T))
    end

    return inv(max(one(T), -λ))
end

function sdpmaxstepdynamic(F::AbstractMatrix{T}, Δx::AbstractVector{T}, scale, work::ConeWorkspace{T}) where {T}
    n = size(F, 1)

    o = 0
    M = reshape(view(work.data, o + 1:o + n * n), n, n); o += n * n
    C = reshape(view(work.data, o + 1:o + n * n), n, n); o += n * n
    α = view(work.data, o + 1:o + n);                    o += n
    β = view(work.data, o + 1:o + n)

    smat!(M, Δx)
    symmetrize!(M)
    mul!(C, M, F)
    mul!(M, F', C)

    if !isnothing(scale)
        @inbounds for j in 1:n, i in 1:n
            M[i, j] /= scale[i] * scale[j]
        end
    end

    @inbounds for j in 1:n
        for i in j:n
            C[i, j] = M[i, j]
        end

        C[j, j] += one(T)
    end

    if issuccess(cholesky!(Symmetric(C, :L); check=false))
        return one(T)
    end

    @inbounds λ = eigminsturm!(M, α, β, -one(T))

    return inv(max(one(T), -λ))
end

############################################################################################
# dualshadow! / primalshadow!
############################################################################################

function dualshadow!(sd::AbstractVector, p::AbstractVector, ::SemidefiniteConeCache, work::ConeWorkspace)
    return sdpshadow!(sd, p, work)
end

function primalshadow!(sp::AbstractVector, d::AbstractVector, ::SemidefiniteConeCache, work::ConeWorkspace)
    return sdpshadow!(sp, d, work)
end

function sdpshadow!(s::AbstractVector{T}, x::AbstractVector{T}, work::ConeWorkspace{T}) where {T}
    d = triroot(length(x))

    M = reshape(view(work.data, 0d^2 + 1:1d^2), d, d)
    Z = reshape(view(work.data, 1d^2 + 1:2d^2), d, d)

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

function sdpshadowstatic!(s::AbstractVector{T}, x::AbstractVector{T}, M::AbstractMatrix{T}, Z::AbstractMatrix{T}, n::Val{N}) where {T, N}
    @inbounds smatstatic!(M, x, n)
    @inbounds cholstatic!(Symmetric(M, :L), n) || return false

    @inbounds for j in 1:N
        for i in 1:N
            Z[i, j] = ifelse(i == j, one(T), zero(T))
        end
    end

    @inbounds ldivstatic!(LowerTriangular(M), Z, n)
    @inbounds ldivstatic!(LowerTriangular(M)', Z, n)
    @inbounds svecstatic!(s, Z, n)
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

############################################################################################
# primalhess!
############################################################################################

function primalhess!(r::AbstractVector, p::AbstractVector, Δp::AbstractVector, cache::SemidefiniteConeCache, work::ConeWorkspace)
    return sdphess!(r, p, Δp, cache.R, work)
end

function sdphess!(r::AbstractVector, p::AbstractVector, Δp::AbstractVector, R::AbstractMatrix, work::ConeWorkspace)
    d = triroot(length(p))
    d == 1 && return sdphessstatic!(r, Δp, R, work, Val(1))
    d == 2 && return sdphessstatic!(r, Δp, R, work, Val(2))
    d == 3 && return sdphessstatic!(r, Δp, R, work, Val(3))
    d == 4 && return sdphessstatic!(r, Δp, R, work, Val(4))
    d == 5 && return sdphessstatic!(r, Δp, R, work, Val(5))
    d == 6 && return sdphessstatic!(r, Δp, R, work, Val(6))
    d == 7 && return sdphessstatic!(r, Δp, R, work, Val(7))
    d == 8 && return sdphessstatic!(r, Δp, R, work, Val(8))
    return sdphessdynamic!(r, Δp, R, work)
end

function sdphessstatic!(r, Δp, R, work, n::Val{N}) where {N}
    m = N * N
    U = reshape(view(work.data, 0m + 1:1m), N, N)
    A = reshape(view(work.data, 1m + 1:2m), N, N)

    @inbounds smatstatic!(U, Δp, n)
    @inbounds symmstatic!(U, n)
    @inbounds mulstatic!(A, R', U, n)
    @inbounds mulstatic!(U, A, R, n)                 # U = Ũ = RᵀUR
    @inbounds mulstatic!(A, R, U, n)
    @inbounds mulstatic!(U, A, R', n)                # U = R Ũ Rᵀ
    @inbounds svecstatic!(r, U, n)
    return r
end

function sdphessdynamic!(r, Δp, R, work)
    n = size(R, 1); m = n * n
    U = reshape(view(work.data, 0m + 1:1m), n, n)
    A = reshape(view(work.data, 1m + 1:2m), n, n)

    smat!(U, Δp)
    symmetrize!(U)
    mul!(A, R', U)
    mul!(U, A, R)                                         # U = Ũ = RᵀUR
    mul!(A, R, U)
    mul!(U, A, R')                                        # U = R Ũ Rᵀ
    svec!(r, U)
    return r
end

############################################################################################
# primalthird!
############################################################################################

function primalthird!(r::AbstractVector, p::AbstractVector, Δp1::AbstractVector, Δp2::AbstractVector, cache::SemidefiniteConeCache, work::ConeWorkspace)
    return sdpthird!(r, p, Δp1, Δp2, cache.R, work)
end

function sdpthird!(r::AbstractVector, p::AbstractVector, Δp1::AbstractVector, Δp2::AbstractVector, R::AbstractMatrix, work::ConeWorkspace)
    d = triroot(length(p))
    d == 1 && return sdpthirdstatic!(r, Δp1, Δp2, R, work, Val(1))
    d == 2 && return sdpthirdstatic!(r, Δp1, Δp2, R, work, Val(2))
    d == 3 && return sdpthirdstatic!(r, Δp1, Δp2, R, work, Val(3))
    d == 4 && return sdpthirdstatic!(r, Δp1, Δp2, R, work, Val(4))
    d == 5 && return sdpthirdstatic!(r, Δp1, Δp2, R, work, Val(5))
    d == 6 && return sdpthirdstatic!(r, Δp1, Δp2, R, work, Val(6))
    d == 7 && return sdpthirdstatic!(r, Δp1, Δp2, R, work, Val(7))
    d == 8 && return sdpthirdstatic!(r, Δp1, Δp2, R, work, Val(8))
    return sdpthirddynamic!(r, Δp1, Δp2, R, work)
end

function sdpthirdstatic!(r::AbstractVector, Δp1::AbstractVector, Δp2::AbstractVector, R::AbstractMatrix, work::ConeWorkspace, n::Val{N}) where {N}
    m = N * N
    U = reshape(view(work.data, 0m + 1:1m), N, N)
    V = reshape(view(work.data, 1m + 1:2m), N, N)
    A = reshape(view(work.data, 2m + 1:3m), N, N)

    @inbounds smatstatic!(U, Δp1, n)
    @inbounds symmstatic!(U, n)
    @inbounds smatstatic!(V, Δp2, n)
    @inbounds symmstatic!(V, n)
    @inbounds mulstatic!(A, R', U, n)
    @inbounds mulstatic!(U, A, R, n)                 # U = Ũ
    @inbounds mulstatic!(A, R', V, n)
    @inbounds mulstatic!(V, A, R, n)                 # V = Ṽ
    @inbounds mulstatic!(A, U, V, n)                 # A = ŨṼ

    @inbounds for i in 1:N                                # A ← ŨṼ + (ŨṼ)ᵀ = ŨṼ + ṼŨ
        A[i, i] *= 2

        for j in i + 1:N
            A[i, j] = A[j, i] = A[i, j] + A[j, i]
        end
    end

    @inbounds mulstatic!(U, R, A, n)
    @inbounds mulstatic!(A, U, R', n)                # A = R(ŨṼ+ṼŨ)Rᵀ
    @inbounds svecstatic!(r, A, n)
    rmul!(r, -1)
    return r
end

function sdpthirddynamic!(r::AbstractVector, Δp1::AbstractVector, Δp2::AbstractVector, R::AbstractMatrix, work::ConeWorkspace)
    n = size(R, 1); m = n * n
    U = reshape(view(work.data, 0m + 1:1m), n, n)
    V = reshape(view(work.data, 1m + 1:2m), n, n)
    A = reshape(view(work.data, 2m + 1:3m), n, n)

    smat!(U, Δp1)
    symmetrize!(U)
    smat!(V, Δp2)
    symmetrize!(V)
    mul!(A, R', U)
    mul!(U, A, R)                                         # U = Ũ
    mul!(A, R', V)
    mul!(V, A, R)                                         # V = Ṽ
    mul!(A, U, V)                                         # A = ŨṼ

    @inbounds for i in 1:n                                # A ← ŨṼ + (ŨṼ)ᵀ = ŨṼ + ṼŨ
        A[i, i] *= 2

        for j in i + 1:n
            A[i, j] = A[j, i] = A[i, j] + A[j, i]
        end
    end

    mul!(U, R, A)
    mul!(A, U, R')                                        # A = R(ŨṼ+ṼŨ)Rᵀ
    svec!(r, A)
    rmul!(r, -1)
    return r
end
