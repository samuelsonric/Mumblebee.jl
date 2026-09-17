"""
    ExponentialCone <: AbstractTDCone

The exponential cone, consisting of all triples (x, y, z)
such that x > 0, y > 0, and y log(x/y) ≥ z.
"""
struct ExponentialCone <: AbstractTDCone end

const ExponentialConeCache{T} = AbstractTDConeCache{ExponentialCone, T}

# compute the fixed point
#
#   f'(e) = -e
#
function expid!(x::AbstractVector)
    x[1] =  1.2909282315382298
    x[2] =  0.8051015526498357
    x[3] = -0.8278379086082098
    return x
end

function identity!(x::AbstractVector, ::ExponentialCone)
    return expid!(x)
end

function initcache!(cache::ExponentialConeCache)
    cache.seed[] = 0.8051015526498357
    return cache
end

# Compute the "determinant"
#
#   det(x) = x₂ log(x₁/x₂) − x₃.
#
# in double precision. This function
# satisfies
#
#   f(x) = -log(det(x)) - log(x₁) - log(x₂).
#
# Returns a quadruple
#
#   (det(x), 1/det(x), log(x₁/x₂), log(x₁/x₂) - 1)
#
function expdet(x1::T, x2::T, x3::T) where {T}
    #
    # lr = log(x₁/x₂), two words (twolograt handles the tiny-lr Sterbenz case and
    # the far case internally, fully relative-accurate at every scale)
    #
    lrh, lrl = twolograt(x1, x2)

    mh, me = twosum(lrh, -one(T))
    lrm1 = mh + (me + lrl)                       # lr − 1 off the two-word lr
    ph, pe = twoprod(lrh, x2)
    pe = muladd(lrl, x2, pe)                     # x₂·lr, two words
    sh, se = twosum(ph, -x3)
    ψh, ψl = twosum(sh, se + pe)
    ih, il = twodiv(one(T), zero(T), ψh, ψl)
    return ψh + ψl, ih + il, lrh + lrl, lrm1
end

# The gradient
#
#   f'(x) = -ψ'(x)/ψ(x) - (1/x₁, 1/x₂, 0)
#
# of the barrier function, from the compensated determinant chain
# (invψ, lrm1).
function expbarrgrad!(g::AbstractVector, x::AbstractVector, invψ, lrm1)
    x1, x2 = x[1], x[2]

    g[1] = -(x2 * invψ + 1) / x1
    g[2] = -lrm1 * invψ - inv(x2)
    g[3] = invψ

    return g
end

# Compute the pivoted Cholesky factor L
#
#   P f''(p) Pᵀ = L Lᵀ.
#
# Returns true on success, false if x is too close to the boundary.
function expfact!(L::AbstractMatrix{T}, x::AbstractVector{T}, ψ, invψ, lrm1) where {T}
    x1, x2 = x[1], x[2]

    r1 = 1 + x2 * invψ           # (ψ + x₂) / ψ
    r2 = (ψ + 2x2) / (ψ + x2)

    flag = ψ > 0 && r1 > 0 && r2 > 0 && isfinite(r1) && isfinite(r2)

    if flag
        s1 = sqrt(r1)

        L[1,1] = invψ
        L[2,1] = -x2 * invψ / x1
        L[3,1] = -lrm1 * invψ
        L[2,2] = s1 / x1
        L[3,2] = -invψ / s1
        L[3,3] = sqrt(r2) / x2

        L[1,2] = zero(T)
        L[1,3] = zero(T)
        L[2,3] = zero(T)
    end

    return flag
end

# Compute the third-order directional derivative
#
#   f'''(x)[u]
#
# as a 3x3 matrix.
function expbarrthird!(D::AbstractMatrix, x::AbstractVector, u::AbstractVector, ψ, invψ, lrm1)
    x1, x2     = x[1], x[2]
    u1, u2, u3 = u[1], u[2], u[3]

    ψg1 = x2 / x1
    ψg2 = lrm1
    ψgu = ψg1 * u1 + ψg2 * u2 - u3

    ψH11 = -x2 / x1^2
    ψH21 =  inv(x1)
    ψH22 = -inv(x2)

    ψHu1 = ψH11 * u1 + ψH21 * u2
    ψHu2 = ψH21 * u1 + ψH22 * u2

    invψ2 = invψ * invψ
    invψ3 = invψ2 * invψ

    α = -2ψgu * invψ3

    D[1,1] =  α * ψg1^2 + (2ψg1 * ψHu1 + ψH11 * ψgu) * invψ2 - (2x2 * u1 / x1^3 - u2 / x1^2) * invψ - 2u1 / x1^3
    D[2,1] =  α * ψg1 * ψg2 + (ψg1 * ψHu2 + ψg2 * ψHu1 + ψH21 * ψgu) * invψ2 + u1 / x1^2 * invψ
    D[3,1] = -α * ψg1 - ψHu1 * invψ2
    D[2,2] =  α * ψg2^2 + (2ψg2 * ψHu2 + ψH22 * ψgu) * invψ2 - u2 / x2^2 * invψ - 2u2 / x2^3
    D[3,2] = -α * ψg2 - ψHu2 * invψ2
    D[3,3] =  α

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

function expboundprim(p::AbstractVector{T}, Δp::AbstractVector{T}) where {T}
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

function expbounddual(d::AbstractVector{T}, Δd::AbstractVector{T}) where {T}
    #
    # z₁ > 0, z₃ < 0
    #
    hi = one(T)

    @inbounds begin
        if Δd[1] < 0
            hi = min(hi, -d[1] / Δd[1])
        end

        if Δd[3] > 0
            hi = min(hi, -d[3] / Δd[3])
        end
    end

    return hi
end

# compute the "shadow" primal, solving
#
#   f'(p*) = -d
#
# via the scale-free normal form
#
#   φ(v) = v + log1p(v) = R
#
# in the reciprocal variable
#
#   v = 1 / (p₂* (-d₃)),
#
# with the constant
#
#   R = h(∞) / (-d₃) = d₂ / (-d₃) - (log(-d₃/d₁) - 1)
#
# computed once. Globally φ' = 1 + 1/(1+v) ∈ (1, 2) and φ is
# concave, and
#
#   0 ≤ log1p(v) ≤ v   ⇒   v* ∈ [R/2, R]
#
# brackets the root for free, so Halley needs no bracketing loop
# and bisection is a one-line safety net (never fired in testing).
# The small-margin regime R → 0 is the well-conditioned one here,
# with v* ≈ R/2; the previous p₂-space iteration degenerated to
# bisection there (20-38 evaluations vs 1-3 for this form).
#
# PRECONDITION: expdualmargin(d) > sqrt(eps(T)).
# Callers must gate on the margin (see tdscale!); the solver stays
# well behaved below the gate, but accuracy degrades as eps/margin
# and the shadow primal is numerically meaningless.
#
function expdualgrad!(sp::AbstractVector{T}, seed::T, d::AbstractVector{T}) where {T}
    @inbounds d1, d2, d3 = d[1], d[2], d[3]

    #
    # d₃ = 0 degenerates h to -1/p₂ + d₂: root p₂ = 1/d₂, no
    # root-find; sp[3] is the d₃ → 0⁻ limit -Inf (the incumbent's
    # inv(d₃) gave the wrong-side +Inf on a positive zero)
    #
    if iszero(d3)
        p2 = inv(d2)
        s  = d2 / d1

        @inbounds sp[1] = s * p2
        @inbounds sp[2] = p2
        @inbounds sp[3] = T(-Inf)

        #
        # closure ray: g(0) = e·d₁, so h(0) = ±Inf by feasibility
        #
        return p2, (d1 >= 0 && d2 >= 0 ? T(Inf) : T(-Inf))
    end

    md3 = -d3
    #
    # compute the constant
    #
    #   R = d₂ / (-d₃) - (log(-d₃/d₁) - 1),
    #
    # splitting the log when the ratio leaves the normal range,
    # and flooring R for graceful out-of-contract behavior (no
    # root exists for R ≤ 0)
    #
    rat = md3 / d1

    if floatmin(T) <= rat < T(Inf)
        lr = log(rat)
    else
        lr = log(md3) - log(d1)
    end

    w = d2 / md3

    R = w - (lr - one(T))

    if !(R > 0)
        R = floatmin(T)
    end
    #
    # bracket for free:
    #
    #   v* ∈ [R/2, R]
    #
    lo = R / 2
    hi = R
    #
    # map the seed into v-space; outside the bracket, start from
    # the series / asymptotic inverse of φ
    #
    #   v₀ = R/2 + R²/16     (R ≤ 2)
    #   v₀ = R - log R       (R > 2)
    #
    # which lies in the bracket for all finite R except when
    # log R < ulp(R) rounds v₀ onto R, where 0.7 R stands in
    #
    v = inv(max(seed, floatmin(T)) * md3)

    if !(lo < v < hi)
        if R <= 2
            v = R / 2 + R^2 / 16
        else
            v = R - log(R)
        end

        if !(lo < v < hi)
            v = T(0.7) * R
        end
    end
    #
    # solve φ(v) = R by one-division Halley; φ'' = -(φ'-1)² is
    # free from φ'
    #
    for _ in 1:60
        fv = v + log1p(v) - R
        fp = one(T) + inv(one(T) + v)

        if fv < 0
            lo = v
        elseif fv > 0
            hi = v
        else
            break
        end

        fpp = -(fp - one(T))^2
        vn  = v - 2fv * fp / (2 * fp^2 - fv * fpp)

        if !(lo < vn < hi)
            #
            # a step that rounds onto a bracket endpoint means
            # the root is within one ulp of it: accept and stop
            #
            if vn == lo || vn == hi
                v = vn
                break
            end

            vn = (lo + hi) / 2
        end
        #
        # the form is scale-free in v, so the stop is purely
        # relative; accuracy of p₂* is then limited only by the
        # conditioning of R (≈ eps/margin, intrinsic)
        #
        converged = abs(vn - v) < 4 * eps(T) * abs(vn)

        v = vn

        if converged
            break
        end
    end
    #
    # recover p* as
    #
    #   p₂* = 1 / ((-d₃) v)
    #   p₁* / p₂* = (-d₃)(v + 1) / d₁
    #   p₃* = p₂* log(p₁*/p₂*) + 1/d₃
    #
    q  = md3 * v
    p2 = inv(q)

    s = md3 * (v + one(T)) / d1

    if !(zero(T) < s < T(Inf))
        s = exp(lr + log1p(v))
    end

    @inbounds sp[1] = s * p2
    @inbounds sp[2] = p2
    @inbounds sp[3] = p2 * log(s) + inv(d3)

    #
    # emit the dual line-search membership value h(0) = 1 − lr + w
    # (w = d₂/(−d₃), already in hand: a pure byproduct)
    #
    return p2, one(T) - lr + w
end

#
# AbstractTDCone Interface
#

function tddet!(cache::ExponentialConeCache, x::AbstractVector)
    @inbounds x1, x2, x3 = x[1], x[2], x[3]
    ψ, invψ, lr, lrm1 = expdet(x1, x2, x3)
    cache.d1[] = ψ
    cache.d2[] = invψ
    cache.d3[] = lr
    cache.d4[] = lrm1
    #
    # the primal line-search membership value h(0) = lr − x₃/x₂; the
    # iterate is strictly interior (x₂ > 0), and an overflowing x₃/x₂
    # yields the semantically correct ±Inf
    #
    @inbounds cache.h0p[] = lr - x3 / x2
    return cache
end

function tdfact!(L::AbstractMatrix, p::AbstractVector, cache::ExponentialConeCache)
    return expfact!(L, p, cache.d1[], cache.d2[], cache.d4[])
end

function tdbarrgrad!(g::AbstractVector, p::AbstractVector, cache::ExponentialConeCache)
    return expbarrgrad!(g, p, cache.d2[], cache.d4[])
end

function tdbarrthird!(D::AbstractMatrix, p::AbstractVector, u::AbstractVector, cache::ExponentialConeCache)
    return expbarrthird!(D, p, u, cache.d1[], cache.d2[], cache.d4[])
end

function tddualgrad!(sp::AbstractVector, seed, d::AbstractVector, cache::ExponentialConeCache)
    next, h0 = expdualgrad!(sp, seed, d)
    cache.h0d[] = h0
    return next
end

function tdboundprim(p::AbstractVector{T}, Δp::AbstractVector{T}, ::ExponentialConeCache) where {T}
    return expboundprim(p, Δp)
end

function tdbounddual(d::AbstractVector{T}, Δd::AbstractVector{T}, ::ExponentialConeCache) where {T}
    return expbounddual(d, Δd)
end

#
# expjetprimlog(τ, p, Δp) -> (h, h′, h″)
#
# the log-form primal membership function h = log(x₁/x₂) − x₃/x₂ along
# the ray x = p + τ Δp. Near the root |x₃/x₂| = |log(x₁/x₂)| ≤ ~745, so
# the absolute error of h is bounded by the universal band; away from
# the root |h| ≫ band and the sign is robust. Closure rays are ±Inf
#

function expjetprimlog(τ::T, p::AbstractVector{T}, Δp::AbstractVector{T}) where {T}
    @inbounds p1, p2, p3 = p[1], p[2], p[3]
    @inbounds Δ1, Δ2, Δ3 = Δp[1], Δp[2], Δp[3]

    x1 = muladd(τ, Δ1, p1)
    x2 = muladd(τ, Δ2, p2)
    x3 = muladd(τ, Δ3, p3)

    if x1 > 0 && x2 > 0
        q = x1 / x2

        if floatmin(T) <= q < T(Inf)
            s = log(q)
        else
            s = log(x1) - log(x2)
        end

        w = x3 / x2

        h = s - w

        wp = (Δ3 - w * Δ2) / x2

        hp = Δ1 / x1 - Δ2 / x2 - wp
        hpp = -(Δ1 / x1)^2 + (Δ2 / x2)^2 + 2 * wp * Δ2 / x2

        return h, hp, hpp
    elseif x1 >= 0 && iszero(x2)
        return (x3 <= 0 ? T(Inf) : T(-Inf)), T(NaN), T(NaN)
    end

    return T(-Inf), T(NaN), T(NaN)
end

#
# expjetduallog(τ, d, Δd) -> (h, h′, h″)
#
# the log-form dual membership function h = 1 + log z₁ − z₂/z₃ − log(−z₃)
# along the ray z = d + τ Δd: the former overflow-strip surrogate, now
# the native global function — finite wherever the domain is, with true
# curvature everywhere
#

function expjetduallog(τ::T, d::AbstractVector{T}, Δd::AbstractVector{T}) where {T}
    @inbounds d1, d2, d3 = d[1], d[2], d[3]
    @inbounds Δ1, Δ2, Δ3 = Δd[1], Δd[2], Δd[3]

    z1 = muladd(τ, Δ1, d1)
    z2 = muladd(τ, Δ2, d2)
    z3 = muladd(τ, Δ3, d3)

    if z1 > 0 && z3 < 0
        r = z2 / z3

        h = one(T) + log(z1) - r - log(-z3)

        rp = (Δ2 - r * Δ3) / z3

        hp = Δ1 / z1 - rp - Δ3 / z3
        hpp = -(Δ1 / z1)^2 + 2 * rp * Δ3 / z3 + (Δ3 / z3)^2

        return h, hp, hpp
    elseif z1 >= 0 && z2 >= 0 && iszero(z3)
        return T(Inf), T(NaN), T(NaN)
    end

    return T(-Inf), T(NaN), T(NaN)
end

function tdjetprimlog(τ::T, p::AbstractVector{T}, Δp::AbstractVector{T}, ::ExponentialConeCache) where {T}
    return expjetprimlog(τ, p, Δp)
end

function tdjetduallog(τ::T, d::AbstractVector{T}, Δd::AbstractVector{T}, ::ExponentialConeCache) where {T}
    return expjetduallog(τ, d, Δd)
end

#
# the direction-dependent halves of the jet(0) seeds; the values h(0)
# live in the cache (h0p, h0d), written where every edge case is
# visible. Pure arithmetic: Inf or NaN from a degenerate division only
# arises when the cached h(0) is ±Inf, which settles the search before
# h′(0) is used
#

function tdhp0prim(p::AbstractVector{T}, Δp::AbstractVector{T}, ::ExponentialConeCache) where {T}
    @inbounds x1, x2, x3 = p[1], p[2], p[3]
    @inbounds Δ1, Δ2, Δ3 = Δp[1], Δp[2], Δp[3]

    w = x3 / x2

    return Δ1 / x1 - Δ2 / x2 - (Δ3 - w * Δ2) / x2
end

function tdhp0dual(d::AbstractVector{T}, Δd::AbstractVector{T}, ::ExponentialConeCache) where {T}
    @inbounds z1, z2, z3 = d[1], d[2], d[3]
    @inbounds Δ1, Δ2, Δ3 = Δd[1], Δd[2], Δd[3]
    r = z2 / z3
    return Δ1 / z1 - (Δ2 - r * Δ3) / z3 - Δ3 / z3
end
