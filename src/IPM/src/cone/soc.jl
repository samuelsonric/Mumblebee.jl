"""
    SecondOrderCone <: AbstractCone

The n + 1-dimensional second-order cone,
consisting of all pairs (x, y) such that
y ≥ ‖x‖.
"""
struct SecondOrderCone <: AbstractCone end

struct SecondOrderConeCache{T} <: AbstractCache{SecondOrderCone}
    cone::SecondOrderCone
    #
    # the square root
    #
    #   β = √det(ω)
    #
    # of the deminant of the
    # Nesterov-Todd scaling point.
    #
    β::FScalarView{T}
    #
    # the normalized Nesterov-Todd scaling
    # point:
    #
    #   w = ω / β
    #
    w::FVectorView{T}
    #
    # the compensated determinants
    #
    #   det(p)
    #   det(d)
    #
    # as (hi, lo) pairs
    #
    pdet::FScalarView{T}
    ddet::FScalarView{T}
    pdetlo::FScalarView{T}
    ddetlo::FScalarView{T}
end

# compute the J-inner product
#
#   xᵀ J y = x₁y₁ - x₂y₂ - … - xₙyₙ
#
# using compensated arithmetic,
# as a (hi, lo) pair.
function socdot(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    @assert length(x) == length(y)
    n = length(x)

    @inbounds s, c = twoprod(x[1], y[1])

    @inbounds for i in 2:n
        p, e = twoprod(x[i], y[i])
        s, e2 = twosum(s, -p)
        c += -e + e2
    end

    return twosum(s, c)
end

# compute the Jordan determinant
#
#   xᵀ J x = x₁x₁ - x₂x₂ - … - xₙxₙ
#
# as a (hi, lo) pair.
function socdet(x::AbstractVector)
    return socdot(x, x)
end

# If flag = false, evaluate the product
#
#   y = √H x.
#
# If flag = true, solve for y:
#
#   √H y = x
#
# where H is the Hessian of the primal barrier
# function at the Nesterov-Todd scaling point.
function socroot!(x::AbstractVector{T}, w::AbstractVector{T}, β::T, flag::Bool) where {T}
    n = length(x)

    if !flag
        σ = -one(T)
        α =  inv(β)
    else
        σ =  one(T)
        α =      β
    end

    @inbounds w1p1 = w[1] + 1

    @inbounds wpdx = w1p1 * x[1]

    @inbounds for i in 2:n
        wpdx = muladd(σ * w[i], x[i], wpdx)
    end

    wpdx /= w1p1

    @inbounds x[1] = α * muladd(wpdx, w1p1, -x[1])

    @inbounds for i in 2:n
        x[i] = α * muladd(σ * wpdx, w[i], x[i])
    end

    return x
end

############################################################################################
# degree
############################################################################################

function degree(::SecondOrderCone, n::Integer)
    return 2
end

############################################################################################
# cachesize
############################################################################################

function cachesize(::Type{SecondOrderCone}, n::Integer)
    return 5 + n   # β, pdet, ddet, pdetlo, ddetlo, then w (n slots)
end

############################################################################################
# workspacesize
############################################################################################

function workspacesize(::Type{SecondOrderCone}, n::Integer)
    return 2n
end

############################################################################################
# cache
############################################################################################

function cache(c::Caches, i::Integer, cone::SecondOrderCone)
    data = cachedata(c, i)
    β     = view(data, 1)
    pdet  = view(data, 2)
    ddet  = view(data, 3)
    pdetlo = view(data, 4)
    ddetlo = view(data, 5)
    w     = view(data, 6:length(data))
    SecondOrderConeCache(cone, β, w, pdet, ddet, pdetlo, ddetlo)
end

############################################################################################
# identity!
############################################################################################

function identity!(x::AbstractVector, ::SecondOrderCone)
    return socid!(x)
end

# construct the identity element
#
#   e = (√2, 0, …, 0)
#
function socid!(x::AbstractVector{T}) where {T}
    fill!(x, zero(T))
    x[1] = sqrt(T(2))
    return x
end

############################################################################################
# scale!
############################################################################################

function scale!(H::AbstractMatrix, p::AbstractVector, d::AbstractVector, cache::SecondOrderConeCache, ::ConeWorkspace)
    flag, β, spsd, pdet, pdetlo, ddet, ddetlo = socscale!(H, cache.w, p, d)
    cache.β[]    = β
    cache.pdet[]  = pdet;  cache.pdetlo[] = pdetlo
    cache.ddet[]  = ddet;  cache.ddetlo[] = ddetlo
    return flag, spsd
end

# Compute the Nesterov-Todd scaling point
# ω in factored form:
#
#   β = √det(ω)
#   w = ω / β
#
# Then assemble the Hessian of the primal
# barrier function evaluated at ω:
#
#   H = (2 (Jw)(Jw)ᵀ - J) / β²
#
function socscale!(
        H::AbstractMatrix{T},
        w::AbstractVector{T},
        p::AbstractVector{T},
        d::AbstractVector{T}
    ) where {T}
    n = length(p)
    #
    # compute the scaling point ω
    #
    pdet, pdetlo = socdet(p)
    ddet, ddetlo = socdet(d)
    #
    # ensure that p and d are strictly interior:
    #
    #   p ∈ ri K
    #   d ∈ ri K
    #
    @inbounds flag = pdet > 0 && p[1] > 0 && ddet > 0 && d[1] > 0

    β = spsd = zero(T)

    if flag
        pdot = cdot(p, d)

        spdet = sqrt(pdet)
        sddet = sqrt(ddet)

        κ = sqrt(two(T) * muladd(spdet, sddet, pdot))
        β = sqrt(spdet / sddet)

        @inbounds w[1] = muladd(d[1], β, p[1] / β) / κ

        @inbounds for i in 2:n
            w[i] = muladd(-d[i], β, p[i] / β) / κ
        end
        #
        # assemble the Hessian
        #
        #   H = (2 (Jw)(Jw)ᵀ - J) / β²
        #
        η = inv(β^2); η2 = 2η

        @inbounds w1 = w[1]

        @inbounds H[1, 1] = muladd(η2, w1^2, H[1, 1] - η)

        nw1 = -η2 * w1

        @inbounds for i in 2:n
            H[i, 1] = H[1, i] = muladd(nw1, w[i], H[1, i])
        end

        @inbounds for j in 2:n
            wj = w[j]

            H[j, j] = muladd(η2, wj^2, H[j, j] + η)

            ew = η2 * wj

            for i in j + 1:n
                H[i, j] = H[j, i] = muladd(ew, w[i], H[j, i])
            end
        end
        #
        # ⟨p*, d*⟩ = 4⟨p, d⟩ / (det(p) det(d))
        #
        spsd = 4pdot / (pdet * ddet)
    end

    return flag, β, spsd, pdet, pdetlo, ddet, ddetlo
end

############################################################################################
# corr!
############################################################################################

function corr!(
        r::AbstractVector,
        p::AbstractVector,
        ::AbstractVector,
        Δp::AbstractVector,
        Δd::AbstractVector,
        σμ::Real,
        cache::SecondOrderConeCache,
        work::ConeWorkspace,
    )
    soccorr!(r, cache.w, cache.β[], p, cache.pdet[], Δp, Δd, σμ, work)
end

function soccorr!(
        r::AbstractVector{T},
        w::AbstractVector{T},
        β::T,
        p::AbstractVector{T},
        pdet::T,
        Δp0::AbstractVector{T},
        Δd0::AbstractVector{T},
        σμ::T,
        work::ConeWorkspace{T},
    ) where {T}
    n = length(p)

    Δp = view(work.data,     1:n)
    Δd = view(work.data, n + 1:n + n)

    copyto!(Δp, Δp0)
    copyto!(Δd, Δd0)

    socroot!(Δp, w, β, false)
    socroot!(Δd, w, β, true)
    #
    # compute the Jordan product
    #
    #   Δp ∘ Δd = (ΔpᵀΔd, Δp₁Δd₂ + Δd₁Δp₂, ..., Δp₁Δdₙ + Δd₁Δpₙ)
    #
    @inbounds Δp1 = Δp[1]
    @inbounds Δd1 = Δd[1]

    @inbounds r[1] = dot(Δp, Δd)

    @inbounds for i in 2:n
        r[i] = muladd(Δd1, Δp[i], Δp1 * Δd[i])
    end

    copyto!(Δp, p)
    socroot!(Δp, w, β, false)
    #
    # solve for x in
    #
    #   Δp ∘ x = r,
    #
    # where Δp ∘ x is the Jordan product
    # of Δp and x.
    #
    Δpdet, _ = socdet(Δp)
    @inbounds Δp1 = Δp[1]

    Δpr, _ = socdot(Δp, r)
    @inbounds r1 = r[1] = Δpr / Δpdet

    iΔp1 = inv(Δp1)

    @inbounds for i in 2:n
        r[i] = muladd(-r1, Δp[i], r[i]) * iΔp1
    end

    axpy!(one(T), Δp, r)
    socroot!(r, w, β, false)

    γ = 2σμ / pdet

    @inbounds r[1] = muladd(γ, p[1], -r[1])

    @inbounds for i in 2:n
        r[i] = muladd(-γ, p[i], -r[i])
    end

    return r
end

############################################################################################
# corr0!
############################################################################################

function corr0!(r::AbstractVector, p::AbstractVector, ::AbstractVector, σμ::Real, cache::SecondOrderConeCache, ::ConeWorkspace)
    return soccorr0!(r, cache.w, cache.β[], p, cache.pdet[], σμ)
end

# soccorr! with Δp = Δd = 0.
function soccorr0!(r::AbstractVector{T}, w::AbstractVector{T}, β::T, p::AbstractVector{T}, pdet::T, σμ::T) where {T}
    n = length(p)

    η = inv(β)^2

    wp, _ = socdot(w, p)
    t = 2η * wp

    c1 = 2σμ / pdet + η

    @inbounds r[1] = muladd(c1, p[1], -t * w[1])

    @inbounds for i in 2:n
        r[i] = muladd(t, w[i], -c1 * p[i])
    end

    return r
end

############################################################################################
# maxsteps
############################################################################################

function maxsteps(p::AbstractVector, Δp::AbstractVector, d::AbstractVector, Δd::AbstractVector, cache::SecondOrderConeCache, ::ConeWorkspace)
    τp = socmaxstep(p, Δp, cache.pdet[], cache.pdetlo[])
    τd = socmaxstep(d, Δd, cache.ddet[], cache.ddetlo[])
    return τp, τd
end

function socmaxstep(x::AbstractVector{T}, Δx::AbstractVector{T}, c::T, clo::T) where {T}
    # compute scalars
    #
    #   a = Δx J Δx
    #   b = 2x J Δx
    #   c =  x J  x
    #
    # such that that
    #
    #   det(x + Δx τ) = (x + Δx τ)ᵀ J (x + Δx τ)
    #                 = aτ² + bτ + c
    #
    a, alo = socdet(Δx)
    b, blo = socdot(x, Δx)
    b *= 2; blo *= 2

    # find the largest number τ ≤ 1 such that
    #
    #   1. a τ² +   b τ + c  ≥ 0
    #   2.        Δx₁ τ + x₁ ≥ 0
    #
    τ = one(T)
    #
    # ensure that
    #
    #   a τ² + b τ + c  ≥ 0
    #
    if abs(a) > eps(T) * max(abs(b), abs(c))
        #
        # d is the discriminant
        #
        #   d = b² - 4ac
        #
        bb, bblo = twoprod(b, b)
        bblo = muladd(2b, blo, bblo)

        ac, aclo = twoprod(4a, c)
        aclo = muladd(4a, clo, muladd(4alo, c, aclo))

        s0, s0lo = twosum(bb, -ac)

        d = s0 + ((bblo - aclo) + s0lo)

        if d > -(eps(T) * max(bb, abs(ac)))
            #
            # - a > 0: roots have the same sign
            #          and τ1 is the smaller one
            #
            # - a < 0: roots have opposite signs
            #          and τ1 is the positive one
            #
            s = sqrt(max(d, zero(T)))
            q = -(b + copysign(s, b)) / 2

            if b ≥ 0
                τ1 = q / a
            else
                τ1 = c / q
            end

            if τ1 > 0
                τ = min(τ, τ1)
            end
        end
    elseif b < zero(T)
        τ = min(τ, -c / b)
    end
    #
    # ensure that
    #
    #   Δx₁ τ + x₁ ≥ 0
    #
    @inbounds Δx1 = Δx[1]

    if Δx1 < 0
        @inbounds τ = min(τ, -x[1] / Δx1)
    end

    return τ
end

############################################################################################
# dualshadow! / primalshadow!
############################################################################################

function dualshadow!(sd::AbstractVector, p::AbstractVector, ::SecondOrderConeCache, ::ConeWorkspace)
    return socshadow!(sd, p)
end

function primalshadow!(sp::AbstractVector, d::AbstractVector, ::SecondOrderConeCache, ::ConeWorkspace)
    return socshadow!(sp, d)
end

# compute the "shadow" iterate
#
#   s = -f'(x) = 2 Jx / det(x),
#
function socshadow!(s::AbstractVector{T}, x::AbstractVector{T}) where {T}
    n = length(x); xdet, _ = socdet(x)

    @inbounds flag = xdet > 0 && x[1] > 0

    if flag
        γ = 2 / xdet

        @inbounds s[1] = γ * x[1]

        @inbounds for i in 2:n
            s[i] = -γ * x[i]
        end
    end

    return flag
end

############################################################################################
# primalhess!
############################################################################################

function primalhess!(r::AbstractVector, p::AbstractVector, Δp::AbstractVector, cache::SecondOrderConeCache, work::ConeWorkspace)
    return sochess!(r, p, Δp, cache.pdet[])
end

function sochess!(r::AbstractVector, p::AbstractVector, Δp::AbstractVector, pdet::Real)
    n = length(p)
    pΔp, _ = socdot(p, Δp)
    pdet2 = pdet^2

    @inbounds r[1] = muladd(4pΔp, p[1], -2pdet * Δp[1]) / pdet2

    @inbounds for i in 2:n
        r[i] = muladd(-4pΔp, p[i], 2pdet * Δp[i]) / pdet2
    end

    return r
end

############################################################################################
# primalthird!
############################################################################################

function primalthird!(r::AbstractVector, p::AbstractVector, Δp1::AbstractVector, Δp2::AbstractVector, cache::SecondOrderConeCache, work::ConeWorkspace)
    return socthird!(r, p, Δp1, Δp2, cache.pdet[])
end

function socthird!(r::AbstractVector, p::AbstractVector, Δp1::AbstractVector, Δp2::AbstractVector, pdet::Real)
    n = length(p)
    pΔp1,   _ = socdot(p, Δp1)
    pΔp2,   _ = socdot(p, Δp2)
    Δp1Δp2, _ = socdot(Δp1, Δp2)

    pdet2 = pdet^2; f = 4 / pdet2; e = 16pΔp1 * pΔp2 / (pdet2 * pdet)

    @inbounds r[1] = muladd(f, muladd(pΔp1, Δp2[1], muladd(pΔp2, Δp1[1], Δp1Δp2 * p[1])), -e * p[1])

    @inbounds for i in 2:n
        r[i] = muladd(-f, muladd(pΔp1, Δp2[i], muladd(pΔp2, Δp1[i], Δp1Δp2 * p[i])), e * p[i])
    end

    return r
end
