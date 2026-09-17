"""
    PowerCone{T} <: AbstractTDCone

The three-dimensional power cone with parameter α ∈ (0, 1),
consisting of all triples (x₁, x₂, x₃) such that
x₁ ≥ 0, x₂ ≥ 0, and x₁^α x₂^(1-α) ≥ |x₃|.
"""
struct PowerCone{T} <: AbstractTDCone
    α::T

    function PowerCone{T}(α) where {T}
        @assert 0 < α < 1
        return new{T}(α)
    end
end

function PowerCone(α::T) where {T}
    return PowerCone{T}(α)
end

const PowerConeCache{T} = AbstractTDConeCache{PowerCone{T}, T}

# construct the identity element
#
#   e = (√(1 + α), √(2 - α), 0)
#
function identity!(x::AbstractVector, cone::PowerCone)
    α = cone.α
    x[1] = sqrt(1 + α)
    x[2] = sqrt(2 - α)
    x[3] = false
    return x
end

function initcache!(cache::PowerConeCache{T}) where {T}
    cache.seed[] = true
    return cache
end

# Compute the "determinant"
#
#   det(x) = x₁²ᵃ x₂²⁻²ᵃ - x₃²
#
# in double precision. This function
# satisfies
#
#   f(x) = -log(det(x)) - (1 - α) log(x₁) - α log(x₂).
#
# Returns a triple
#
#   (x₁ᵃx₂ᵇ, x₁ᵃx₂ᵇ / det(x), 1 / det(x)).
#
function powdet(x1::T, x2::T, x3::T, α::T) where {T}
    a = 2α                                        # exact; uses the stored α
    #
    # P = x₁^a x₂^b = 2^L2,  b = 2 − a,  regrouped in base 2 so b never appears:
    #   L2 = a·log₂(x₁/x₂) + 2·log₂x₂
    #
    rh, rl = twolog2rat(x1, x2)                   # log₂(x₁/x₂), two words
    gh, gl = DF._unsafe_log2((x2, zero(T)))       # log₂x₂, two words
    L2 = DF.add_dddd_dd_(DF.mul_dddd_dd_((a, zero(T)), (rh, rl)), (2gh, 2gl))
    Ph, Pl = twoexp2(L2[1], L2[2])
    qh, ql = twoprod(x3, x3)
    fh, fe = twosum(Ph, -qh)
    dh, dl = twosum(fh, fe + (Pl - ql))           # det, two words
    ih, il = twodiv(one(T), zero(T), dh, dl)
    ρh, ρe = twoprod(Ph, ih)                      # ρ = P·invdet
    ρe = muladd(Ph, il, muladd(Pl, ih, ρe))
    ρh, ρl = twosum(ρh, ρe)
    return Ph + Pl, ρh + ρl, ih + il
end

# evaluate the gradient
#
#   f'(p)
#
# of the barrier function f, from the compensated determinant chain
# (ρ, invdet).
function powbarrgrad!(g::AbstractVector, x::AbstractVector, α, ρ, invdet)
    a = 2α
    b = 2 - 2α

    g[1] = -(a * ρ + 1 - α) / x[1]
    g[2] = -(b * ρ     + α) / x[2]
    g[3] =  2x[3] * invdet

    return g
end

# factorize the Hessian
#
#   P f''(p) Pᵀ = L Lᵀ
#
# of the barrier function f, using the pivot order
# (3, 1, 2). The factor is built directly from p;
# the Hessian is never assembled.
#
# Returns true on success, false if the Schur complements
# are non-positive (p too close to the boundary).
function powfact!(L::AbstractMatrix, x::AbstractVector, α, p, ρ, invdet)
    x1, x2, x3 = x[1], x[2], x[3]

    a = 2α
    b = 2 - 2α

    w = ρ * invdet
    s = p + x3 * x3

    l1 = a / x1
    l2 = b / x2

    d1 = (2ρ * a + b) / 2x1^2
    d2 = (2ρ * b + a) / 2x2^2
    c  = ρ * x3^2 / s

    r1 =  d1      - c *  l1^2
    r2 = (d1 * d2 - c * (l2^2 * d1 + l1^2 * d2)) / r1

    flag = r1 > 0 && r2 > 0 && isfinite(r2)

    if flag
        L[1,1] = sqrt(2s) * invdet       # √H₃₃
        L[2,1] = -2l1 * x3 * w / L[1,1]  # H₁₃ / L₁₁
        L[3,1] = -2l2 * x3 * w / L[1,1]  # H₂₃ / L₁₁
        L[2,2] = sqrt(r1)
        L[3,3] = sqrt(r2)
        L[3,2] = -c * l1 * l2 / L[2,2]

        L[1,2] = false
        L[1,3] = false
        L[2,3] = false
    end

    return flag
end

# compute the third-order directional derivative
#
#   f'''(p)[u]
#
# as a 3x3 matrix
function powbarrthird!(D::AbstractMatrix, x::AbstractVector, u::AbstractVector, α, p, ρ, invdet)
    x1, x2, x3 = x[1], x[2], x[3]
    u1, u2, u3 = u[1], u[2], u[3]

    a = 2α
    b = 2 - 2α

    l1 = a / x1
    l2 = b / x2
    m1 = u1 / x1
    m2 = u2 / x2

    φp1 = p * l1
    φp2 = p * l2
    φp3 = -2x3

    φdot = φp1 * u1 + φp2 * u2 + φp3 * u3

    φpp11 = p * l1 * (l1 - inv(x1))
    φpp22 = p * l2 * (l2 - inv(x2))
    φpp12 = p * l1 * l2

    φdotp1 = φpp11 * u1 + φpp12 * u2
    φdotp2 = φpp12 * u1 + φpp22 * u2
    φdotp3 = -2u3

    invdet2 = invdet * invdet
    invdet3 = invdet2 * invdet

    c2 =  φdot * invdet2
    c3 = 2φdot * invdet3

    K  = ρ * a * b * (a - 1) * (m2 - m1)

    D[1,1] = 2φdotp1 * φp1                 * invdet2 - c3 * φp1 * φp1 - K / (x1 * x1) +  c2 * φpp11 - b * m1 / (x1 * x1)
    D[2,2] = 2φdotp2 * φp2                 * invdet2 - c3 * φp2 * φp2 - K / (x2 * x2) +  c2 * φpp22 - a * m2 / (x2 * x2)
    D[2,1] = (φdotp2 * φp1 + φp2 * φdotp1) * invdet2 - c3 * φp2 * φp1 + K / (x1 * x2) +  c2 * φpp12
    D[3,3] = 2φdotp3 * φp3                 * invdet2 - c3 * φp3 * φp3                 - 2c2
    D[3,1] = (φdotp3 * φp1 + φp3 * φdotp1) * invdet2 - c3 * φp3 * φp1
    D[3,2] = (φdotp3 * φp2 + φp3 * φdotp2) * invdet2 - c3 * φp3 * φp2

    D[1,2] = D[2,1]
    D[1,3] = D[3,1]
    D[2,3] = D[3,2]

    return D
end

# Compute the largest step to the domain boundary,
#
#   hi = min(1, first affine clamp of each strictly-signed
#             domain coordinate),
#
# for each cone's open coordinates.

function powboundprim(p::AbstractVector{T}, Δp::AbstractVector{T}) where {T}
    #
    # x₁ > 0, x₂ > 0
    #
    hi = one(T)

    @inbounds begin
        if Δp[1] < 0
            hi = min(hi, -p[1] / Δp[1])
        end

        if Δp[2] < 0
            hi = min(hi, -p[2] / Δp[2])
        end
    end

    return hi
end

function powbounddual(d::AbstractVector{T}, Δd::AbstractVector{T}) where {T}
    #
    # s₁ > 0, s₂ > 0
    #
    hi = one(T)

    @inbounds begin
        if Δd[1] < 0
            hi = min(hi, -d[1] / Δd[1])
        end

        if Δd[2] < 0
            hi = min(hi, -d[2] / Δd[2])
        end
    end

    return hi
end

# compute the primal "shadow" iterate p*, solving
#
#   -f'(p*) = d,
#
# via the scale-free normal form
#
#   1 - u = W₀ (1 + k₁u)ᵃ (1 + k₂u)ᵇ
#
# in the reciprocal variable u = 1/ρ ∈ (0, 1], with
#
#   a = 2α,  b = 2 - 2α,  k₁ = (1-α)/a,  k₂ = α/b
#
# and the constant
#
#   W₀ = ¼ d₃² (a/d₁)ᵃ (b/d₂)ᵇ
#
# computed once. W₀ < 1 is exactly dual-cone membership and
# f(0) = 1 - W₀ is the margin; on [0, u*] the increasing factor
# W stays below 1 and k₁a + k₂b = 1, so
#
#   |f'| ∈ (1, 2)   ⇒   u* ∈ [(1-W₀)/2, 1-W₀]
#
# brackets the root for free — the same slope window and bracket
# structure as the exp normal form. Every per-evaluation quantity
# is O(1) whatever the scales of d (all magnitude sits in W₀),
# with one pow per evaluation at the exact exponent a = 2α; the
# two-pow form with b̃ = fl(2-2α) carries an exponent bias and
# raced 15-30% slower.
#
# PRECONDITION: powdualmargin(d, α) > sqrt(eps(T)).
# Callers must gate on the margin (see tdscale!); the solver stays
# well behaved below the gate, but accuracy degrades as eps/margin
# and the shadow primal is numerically meaningless.
#
function powdualgrad!(sp::AbstractVector{T}, seed::T, d::AbstractVector{T}, α::T) where {T}
    @inbounds d1, d2, d3 = d[1], d[2], d[3]

    a =     2α
    b = 2 - 2α
    #
    # d₃ = 0 gives ρ = 1 and p₃* = 0 exactly: no root-find
    #
    if iszero(d3)
        @inbounds sp[1] = (a + 1 - α) / d1
        @inbounds sp[2] = (b +     α) / d2
        @inbounds sp[3] = zero(T)

        #
        # d₃ = 0 ray: t₁ > 0 at an interior dual, so h(0) = +Inf
        #
        return one(T), T(Inf)
    end

    k1 = (1 - α) / a
    k2 =      α  / b
    t1 = a / d1
    t2 = b / d2
    #
    # compute W₀ once, in the balanced grouping
    #
    #   W₀ = ((d₃/2) t₂)² (t₁/t₂)ᵃ,
    #
    # requiring every intermediate to be NORMAL: a subnormal
    # intermediate (d₃² for |d₃| < ~1e-154, or the ratio at
    # extreme d₁/d₂) quantizes catastrophically yet can launder
    # into a normal-looking W₀. Otherwise fall back to log space,
    # whose Σ|log| eps rounding matters only beyond ~1e±150
    #
    z1  = (d3 / 2) * t2
    z1s = z1 * z1
    z2  = t1 / t2
    z2a = z2^a
    W0  = z1s * z2a

    normal = floatmin(T) <= z1s < T(Inf) &&
             floatmin(T) <= z2  < T(Inf) &&
             floatmin(T) <= z2a < T(Inf) &&
             floatmin(T) <= W0  < T(Inf)

    if !normal
        lW0 = 2log(abs(d3)) - log(T(4)) +
              a * (log(a) - log(d1)) + b * (log(b) - log(d2))
        W0 = exp(lW0)
    else
        #
        # log W₀ is now the solver's own working constant (the residual
        # below is log-form) and doubles as the dual jet(0) seed:
        # h(0) = −log(W₀)/2 exactly. On this path its absolute error is
        # ~eps; the fallback's ~350ε also certifies against the seed's
        # constant band
        #
        lW0 = log(W0)
    end
    #
    # bracket for free, flooring the margin f(0) = 1 - W₀ for
    # graceful out-of-contract behavior (no root for W₀ ≥ 1):
    #
    #   u* ∈ [f₀/2, f₀]
    #
    f0 = 1 - W0

    if !(f0 > 0)
        f0 = floatmin(T)
    end

    lo = f0 / 2
    hi = f0
    #
    # map the seed into u-space; outside the bracket, take a free
    # Halley step from u = 0 built from the precomputed constants
    #
    #   f(0)   = f₀
    #   f'(0)  = -(1 + W₀)
    #   f''(0) = -W₀ (1 - k₁²a - k₂²b),
    #
    # which lands in the bracket except at extreme α, where the
    # Halley denominator changes sign and 0.7 f₀ stands in
    #
    u = inv(max(seed, one(T)))

    if !(lo < u < hi)
        Fpp0 = one(T) - (k1^2 * a + k2^2 * b)

        u = -4lW0 / (8 - lW0 * Fpp0)

        if !(lo < u < hi)
            u = T(0.7) * f0
        end
    end
    #
    # solve the LOG residual
    #
    #   F(u) = log W₀ + a log1p(k₁u) + b log1p(k₂u) − log1p(−u) = 0
    #
    # (same root, measured equal-or-better accuracy at every margin,
    # since F's absolute error is ~eps from lW0 versus the linear
    # residual's ~4ε per-iteration W). The step is regime-selected:
    # where the 1/(1−u) pole dominates the slope, Halley-in-u creeps,
    # and the pole-matched step 1 − (1−u)·exp(F) — exact for
    # F ≈ A − log(1−u) — is primary; in the margin regime (F' ≈ 2)
    # Halley is primary. Each safeguards the other, then bisection
    #
    for _ in 1:60
        r1 = 1 + k1 * u
        r2 = 1 + k2 * u
        Fv = lW0 + a * log1p(k1 * u) + b * log1p(k2 * u) - log1p(-u)

        σ  = k1 * a / r1 + k2 * b / r2
        Fp = σ + 1 / (1 - u)

        if Fv < 0
            lo = u
        elseif Fv > 0
            hi = u
        else
            break
        end

        σp  = -(k1^2 * a / r1^2 + k2^2 * b / r2^2)
        Fpp = σp + 1 / (1 - u)^2

        if 1 / (1 - u) > 4σ
            un  = 1 - (1 - u) * exp(Fv)
            alt = u - 2Fv * Fp / (2 * Fp^2 - Fv * Fpp)
        else
            un  = u - 2Fv * Fp / (2 * Fp^2 - Fv * Fpp)
            alt = 1 - (1 - u) * exp(Fv)
        end

        if !(lo < un < hi)
            #
            # a step that rounds onto a bracket endpoint means
            # the root is within one ulp of it: accept and stop
            #
            if un == lo || un == hi
                u = un
                break
            end

            un = (lo < alt < hi) ? alt : (lo + hi) / 2
        end
        #
        # F carries ~eps absolute (from lW0), so u is resolvable
        # to ~eps absolute — the eps/margin conditioning of
        # ρ = 1/u, intrinsic to the problem in any parametrization
        #
        converged = abs(un - u) < 4 * eps(T) * (1 + abs(un))

        u = un

        if converged
            break
        end
    end

    ρ = inv(u)
    #
    # recover p* as
    #
    #   p₁* = (aρ + 1 - α) / d₁
    #   p₂* = (bρ +     α) / d₂
    #   p₃* = -2(ρ - 1) / d₃
    #
    @inbounds sp[1] = (a * ρ + 1 - α) / d1
    @inbounds sp[2] = (b * ρ +     α) / d2
    @inbounds sp[3] = -2(ρ - one(T)) / d3

    #
    # emit the dual line-search membership value h(0) = −log(W₀)/2;
    # valid on both W₀ paths (the fallback's ~350ε certifies against
    # the constant band)
    #
    return ρ, -lW0 / 2
end

#
# AbstractTDCone Interface
#

function tddet!(cache::PowerConeCache, p::AbstractVector)
    @inbounds p1, p2, p3 = p[1], p[2], p[3]
    pv, ρ, invdet = powdet(p1, p2, p3, cache.cone.α)
    cache.d1[] = pv
    cache.d2[] = ρ
    cache.d3[] = invdet
    #
    # the primal line-search membership value h(0) = log(P)/2 − log|x₃|;
    # x₃ = 0, and P over/underflow, all land as the semantically correct
    # ±Inf (NaN in the doubly-degenerate corner reads as not-certified)
    #
    @inbounds cache.h0p[] = log(pv) / 2 - log(abs(p3))
    return cache
end

function tdfact!(L::AbstractMatrix, p::AbstractVector, cache::PowerConeCache{T}) where {T}
    return powfact!(L, p, cache.cone.α, cache.d1[], cache.d2[], cache.d3[])
end

function tdbarrgrad!(g::AbstractVector, p::AbstractVector, cache::PowerConeCache{T}) where {T}
    return powbarrgrad!(g, p, cache.cone.α, cache.d2[], cache.d3[])
end

function tdbarrthird!(D::AbstractMatrix, p::AbstractVector, u::AbstractVector, cache::PowerConeCache{T}) where {T}
    return powbarrthird!(D, p, u, cache.cone.α, cache.d1[], cache.d2[], cache.d3[])
end

function tddualgrad!(sp::AbstractVector, seed, d::AbstractVector, cache::PowerConeCache{T}) where {T}
    next, h0 = powdualgrad!(sp, seed, d, cache.cone.α)
    cache.h0d[] = h0
    return next
end

function tdboundprim(p::AbstractVector{T}, Δp::AbstractVector{T}, ::PowerConeCache{T}) where {T}
    return powboundprim(p, Δp)
end

function tdbounddual(d::AbstractVector{T}, Δd::AbstractVector{T}, ::PowerConeCache{T}) where {T}
    return powbounddual(d, Δd)
end

#
# powjetprimlog(τ, p, Δp, α) -> (h, h′, h″)
#
# the log-form primal membership function
# h = α log x₁ + β log x₂ − log|x₃| along the ray x = p + τ Δp; every
# log term is bounded by ~745, bounding the absolute error of h by the
# universal band, and β enters linearly (no exponent bias). Closure
# rays are ±Inf
#

function powjetprimlog(τ::T, p::AbstractVector{T}, Δp::AbstractVector{T}, α::T) where {T}
    @inbounds p1, p2, p3 = p[1], p[2], p[3]
    @inbounds Δ1, Δ2, Δ3 = Δp[1], Δp[2], Δp[3]

    β = one(T) - α

    x1 = muladd(τ, Δ1, p1)
    x2 = muladd(τ, Δ2, p2)
    x3 = muladd(τ, Δ3, p3)

    if x1 > 0 && x2 > 0
        if iszero(x3)
            return T(Inf), T(NaN), T(NaN)
        end

        h = α * log(x1) + β * log(x2) - log(abs(x3))

        hp = α * Δ1 / x1 + β * Δ2 / x2 - Δ3 / x3
        hpp = -α * (Δ1 / x1)^2 - β * (Δ2 / x2)^2 + (Δ3 / x3)^2

        return h, hp, hpp
    elseif (iszero(x1) && x2 >= 0) || (x1 >= 0 && iszero(x2))
        return (iszero(x3) ? zero(T) : T(-Inf)), T(NaN), T(NaN)
    end

    return T(-Inf), T(NaN), T(NaN)
end

#
# powjetduallog(τ, d, Δd, α) -> (h, h′, h″)
#
# the log-form dual membership function along s = d + τ Δd, with
# u₁ = s₁/α and u₂ = s₂/β
#

function powjetduallog(τ::T, d::AbstractVector{T}, Δd::AbstractVector{T}, α::T) where {T}
    @inbounds d1, d2, d3 = d[1], d[2], d[3]
    @inbounds Δ1, Δ2, Δ3 = Δd[1], Δd[2], Δd[3]

    β = one(T) - α

    s1 = muladd(τ, Δ1, d1)
    s2 = muladd(τ, Δ2, d2)
    s3 = muladd(τ, Δ3, d3)

    if s1 > 0 && s2 > 0
        if iszero(s3)
            return T(Inf), T(NaN), T(NaN)
        end

        h = α * log(s1 / α) + β * log(s2 / β) - log(abs(s3))

        hp = α * Δ1 / s1 + β * Δ2 / s2 - Δ3 / s3
        hpp = -α * (Δ1 / s1)^2 - β * (Δ2 / s2)^2 + (Δ3 / s3)^2

        return h, hp, hpp
    elseif (iszero(s1) && s2 >= 0) || (s1 >= 0 && iszero(s2))
        return (iszero(s3) ? zero(T) : T(-Inf)), T(NaN), T(NaN)
    end

    return T(-Inf), T(NaN), T(NaN)
end

function tdjetprimlog(τ::T, p::AbstractVector{T}, Δp::AbstractVector{T}, cache::PowerConeCache{T}) where {T}
    return powjetprimlog(τ, p, Δp, cache.cone.α)
end

function tdjetduallog(τ::T, d::AbstractVector{T}, Δd::AbstractVector{T}, cache::PowerConeCache{T}) where {T}
    return powjetduallog(τ, d, Δd, cache.cone.α)
end

function tdhp0prim(p::AbstractVector{T}, Δp::AbstractVector{T}, cache::PowerConeCache{T}) where {T}
    @inbounds x1, x2, x3 = p[1], p[2], p[3]
    @inbounds Δ1, Δ2, Δ3 = Δp[1], Δp[2], Δp[3]

    α = cache.cone.α

    return α * Δ1 / x1 + (one(T) - α) * Δ2 / x2 - Δ3 / x3
end

function tdhp0dual(d::AbstractVector{T}, Δd::AbstractVector{T}, cache::PowerConeCache{T}) where {T}
    @inbounds s1, s2, s3 = d[1], d[2], d[3]
    @inbounds Δ1, Δ2, Δ3 = Δd[1], Δd[2], Δd[3]
    α = cache.cone.α
    return α * Δ1 / s1 + (one(T) - α) * Δ2 / s2 - Δ3 / s3
end
