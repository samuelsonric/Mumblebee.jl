@propagate_inbounds function svdjacobi!(U::AbstractMatrix{T}, σ::AbstractVector{T}) where {T}
    n = size(U, 1)

    @boundscheck checkbounds(U, n, n)
    @boundscheck checkbounds(σ, n)
    #
    # compute the singular-value decomposition
    #
    #   A = U Σ Vᵀ
    #
    # by applying Jacobi rotations to the
    # columns of A until they are orthogonal:
    #
    #   A J₁ J₂ ⋯ = U Σ
    #
    # compute the norm
    #
    #   nA := ‖ A ‖∞
    #
    #       = max |Aᵢⱼ|
    #         i,j
    #
    nA = zero(T)

    @inbounds for j in 1:n
        for i in 1:n
            nA = max(nA, abs(U[i, j]))
        end
    end

    done = false

    #
    # rescale A by a power of two so that
    #
    #   ‖ A ‖∞ ≈ 1
    #
    eA = exponent(nA)
    eA1 = eA >> 1
    r1 = ldexp(one(T),      - eA1)
    r2 = ldexp(one(T), -(eA - eA1))

    @inbounds for j in 1:n
        for i in 1:n
            U[i, j] *= r1
            U[i, j] *= r2
        end
    end
    #
    # visit the columns of A in odd-even ordering
    #
    #   (1,2) (3,4) ⋯
    #   (2,3) (4,5) ⋯
    #
    # exhanging each pair after it is processed
    #
    iter = 0

    @inbounds while !done && iter < 30
        iter += 1
        done = true
        #
        # compute the squared column norms
        #
        #   dⱼ := ‖ aⱼ ‖²
        #
        for j in 1:n
            d = zero(T)

            @simd for i in 1:n
                d = muladd(U[i, j], U[i, j], d)
            end

            σ[j] = d
        end

        for k in 1:n
            if isodd(k)
                strt = 1
            else
                strt = 2
            end

            for p in strt:2:n - 1
                q = p + 1
                #
                # compute the inner product
                #
                #   g := aₚᵀ aₚ₊₁
                #
                g = zero(T)

                @simd for i in 1:n
                    g = muladd(U[i, p], U[i, q], g)
                end

                dp = σ[p]
                dq = σ[q]
                #
                # if aₚ and aₚ₊₁ are not orthogonal to
                # working precision
                #
                #   g² > n ε² dₚ dₚ₊₁
                #
                # then apply a Jacobi rotation J making
                # them so — the identity off the (p, q)
                # plane, a pure rotation there,
                #                  p  q
                #       [ 1                      ]
                #       [    ⋱                   ]
                #       [       1                ]
                #   J = [          c  s          ] p ,   q = p + 1
                #       [         −s  c          ] q
                #       [                1       ]
                #       [                   ⋱    ]
                #       [                      1 ]
                #
                # where
                #
                #   c = δ / √(2 ω |δ|),  ζ = (dₚ₊₁ − dₚ) / 2,  ω = √(ζ² + g²),  δ = { ζ + ω if ζ ≥ 0
                #   s = g / √(2 ω |δ|)                                              { ζ - ω if ζ < 0
                #
                # finally, exchange the two columns
                #
                if g * g > n * eps(T)^2 * dp * dq
                    done = false

                    ζ = (dq - dp) / 2
                    ω = sqrt(muladd(ζ, ζ, g * g))
                    δ = ζ + copysign(ω, ζ)
                    ν = inv(sqrt(2 * ω * abs(δ)))
                    c = δ * ν
                    s = g * ν
                    t = g / δ

                    @simd for i in 1:n
                        bp = U[i, p]
                        bq = U[i, q]

                        U[i, p] = muladd(s, bp,  c * bq)
                        U[i, q] = muladd(c, bp, -s * bq)
                    end

                    σ[p] = muladd( t, g, dq)
                    σ[q] = muladd(-t, g, dp)
                else
                    @simd for i in 1:n
                        U[i, p], U[i, q] = U[i, q], U[i, p]
                    end

                    σ[p] = dq
                    σ[q] = dp
                end
            end
        end

    end
    #
    # finalize σ and U:
    #
    #   σⱼ ← √σⱼ,  uⱼ ← uⱼ / √σⱼ
    #
    @inbounds for j in 1:n
        nrm = sqrt(σ[j])
        σ[j] = ldexp(nrm, eA)
        r = inv(nrm)

        for i in 1:n
            U[i, j] *= r
        end
    end

    return done
end

# ============================================================================
# Analytic eigmin for small symmetric matrices
# ============================================================================

@propagate_inbounds function eigminstatic(M::AbstractMatrix, ::Val{1})
    @boundscheck checkbounds(M, 1, 1)
    @inbounds return M[1,1]
end

@propagate_inbounds function eigminstatic(M::AbstractMatrix{T}, ::Val{2}) where T
    @boundscheck checkbounds(M, 2, 2)
    @inbounds M₁₁ = M[1,1]
    @inbounds M₂₂ = M[2,2]
    @inbounds M₂₁ = M[2,1]

    h =       (M₁₁ + M₂₂) / 2
    r = hypot((M₁₁ - M₂₂) / 2, M₂₁)

    if h ≥ 0
        λmax = h + r
    else
        λmax = h - r
    end

    if iszero(λmax)
        λmin = λmax
    else
        λmin = min(λmax, muladd(M₁₁, M₂₂, -M₂₁ * M₂₁) / λmax)
    end

    return λmin
end

@propagate_inbounds function eigminstatic(M::AbstractMatrix{T}, ::Val{3}) where T
    @boundscheck checkbounds(M, 3, 3)
    @inbounds M₁₁ = M[1,1]
    @inbounds M₂₂ = M[2,2]
    @inbounds M₃₃ = M[3,3]
    @inbounds M₂₁ = M[2,1]
    @inbounds M₃₁ = M[3,1]
    @inbounds M₃₂ = M[3,2]

    m = max(abs(M₁₁), abs(M₂₂), abs(M₃₃), abs(M₂₁), abs(M₃₁), abs(M₃₂))

    if iszero(m)
        return zero(T)
    elseif !isfinite(m)
        return T(NaN)
    else
        e = exponent(m)
        s = ldexp(one(T), -e)

        B₁₁ = M₁₁ * s; B₂₂ = M₂₂ * s; B₃₃ = M₃₃ * s
        B₂₁ = M₂₁ * s; B₃₁ = M₃₁ * s; B₃₂ = M₃₂ * s

        q = (B₁₁ + B₂₂ + B₃₃) / 3

        C₁₁ = B₁₁ - q
        C₂₂ = B₂₂ - q
        C₃₃ = B₃₃ - q

        p = sqrt((C₁₁^2 + C₂₂^2 + C₃₃^2 + 2(B₂₁^2 + B₃₁^2 + B₃₂^2)) / 6)

        if p ≤ 4eps(T) * max(abs(q), one(T))
            r = 4eps(T) * abs(q)
        else
            D₁₁ = C₁₁ / p; D₂₂ = C₂₂ / p; D₃₃ = C₃₃ / p
            D₂₁ = B₂₁ / p; D₃₁ = B₃₁ / p; D₃₂ = B₃₂ / p

            r = ( D₁₁ * (D₂₂ * D₃₃ - D₃₂ * D₃₂)
                - D₂₁ * (D₂₁ * D₃₃ - D₃₂ * D₃₁)
                + D₃₁ * (D₂₁ * D₃₂ - D₂₂ * D₃₁) ) / 2

            r = clamp(r, -one(T), one(T))

            r = 2p * (-cos((acos(r) + π + π) / 3) + 16eps(T) / sqrt(max(one(T) - r^2, 16eps(T))))
        end

        return ldexp(q - r, e)
    end
end

@propagate_inbounds function tridiagonalize!(M::AbstractMatrix{T}, α::AbstractVector{T}, β::AbstractVector{T}) where {T}
    n = size(M, 1)

    @boundscheck checkbounds(M, n, n)
    @boundscheck checkbounds(α, n)
    @boundscheck n < 2 || checkbounds(β, n - 1)
    #
    # compute the Householder tridiagonalization
    #
    #   M = Q T Qᵀ
    #
    # where T is tridiagonal and
    #
    #   Q = H₁ ⋯ Hₙ₋₂
    #
    # is a product of Householder reflections. The
    # diagonal entries of T are written to α and the
    # off-diagonal entries are written to β.
    #
    #   T = [ α₁ β₁       ]
    #       [ β₁ α₂ β₂    ]
    #       [    β₂ α₃ ⋱  ]
    #       [       ⋱  ⋱  ]
    #
    @fastmath @inbounds for k in 1:n - 2
        #
        # The k … n submatrix of M is of the form
        #
        #         k   k + 1 … n
        #   M = [ αₖ  xᵀ ] k
        #       [ x   Y  ] k + 1 … n
        #
        # where x is a vector
        #
        #   x = [ x₁ ] k + 1
        #       [ xᵣ ] k + 2 … n
        #
        # compute the squared norm
        #
        #   nxr := ‖ xᵣ ‖²
        #
        nxr = zero(T)

        for i in k + 2:n
            nxr += M[i, k] * M[i, k]
        end

        x1 = M[k + 1, k]

        if iszero(nxr)
            β[k] = x1
            M[k + 1, k] = zero(T)
        else
            #
            # construct the Householder reflection
            #
            #   H = I - τ v vᵀ,   τ = (βₖ - x₁) / βₖ,  βₖ = { -‖x‖ if x₁ > 0
            #                                               {  ‖x‖ if x₁ < 0
            # where v is the vector
            #
            #   v = (x - βₖ e₁) / (x₁ - βₖ)
            #
            β[k] = βk = -copysign(sqrt(x1 * x1 + nxr), x1)

            M[k + 1, k] = one(T)

            r  = inv(x1 - βk)

            for i in k + 2:n
                M[i, k] *= r
            end

            τ  = (βk - x1) / βk
            #
            # compute the vector
            #
            #   w := z - 1/2τ (zᵀ v) v,   z = τ Y v
            #
            # satisfying
            #
            #   H Y H = Y - v wᵀ - w vᵀ
            #
            c = zero(T)

            for i in k + 1:n
                wi = zero(T)

                for j in k + 1:i
                    wi += M[i, j] * M[j, k]
                end

                for j in i + 1:n
                    wi += M[j, i] * M[j, k]
                end

                α[i] = wi *= τ

                c += wi * M[i, k]
            end

            c *= τ / 2

            for i in k + 1:n
                α[i] -= c * M[i, k]
            end
            #
            # apply Householder reflection to Y:
            #
            #   Y ← H Y H
            #
            for j in k + 1:n
                vj = M[j, k]
                wj = α[j]

                for i in j:n
                    M[i, j] -= M[i, k] * wj + α[i] * vj
                end
            end
        end
    end

    if n ≥ 2
        @inbounds β[n - 1] = M[n, n - 1]
    end

    @inbounds for i in 1:n
        α[i] = M[i, i]
    end

    return α, β
end

@propagate_inbounds function checksturm(α::AbstractVector{T}, β::AbstractVector{T}, mid::T, n::Int) where {T}
    #
    # T - mid I is positive definite
    # if-and-only if its leading principal
    # minors
    #
    #   p₀ =  1
    #   p₁ =  α₁ - mid
    #    ⋮
    #   pᵢ = (αᵢ - mid) pᵢ₋₁ - βᵢ₋₁² pᵢ₋₂
    #    ⋮
    #   pₙ = (αₙ - mid) pₙ₋₁ - βₙ₋₁² pₙ₋₂
    #
    # are all positive. If any pᵢ is nonpositive,
    # then λ < mid.
    #
    flag = false

    pm1 = one(T); p = α[1] - mid

    if !(p > 0)
        flag = true
    else
        for i in 2:n
            pp1 = muladd(α[i] - mid, p, -β[i - 1]^2 * pm1)

            if !(pp1 > 0)
                flag = true
                break
            end

            pm1 = p; p = pp1
        end
    end

    return flag
end

@propagate_inbounds function eigminsturm!(M::AbstractMatrix{T}, α::AbstractVector{T}, β::AbstractVector{T}, tol::T, λmax::T) where {T}
    n = size(M, 1)

    @boundscheck checkbounds(M, n, n)
    @boundscheck checkbounds(α, n)
    @boundscheck n < 2 || checkbounds(β, n - 1)
    #
    # compute the Householder tridiagonalization
    #
    #   M = Q T Qᵀ
    #
    # where T is tridiagonal and
    #
    #   Q = H₁ ⋯ Hₙ₋₂
    #
    # is a product of Householder reflections. The
    # diagonal entries of T are written to α and the
    # off-diagonal entries are written to β.
    #
    #   T = [ α₁ β₁       ]
    #       [ β₁ α₂ β₂    ]
    #       [    β₂ α₃ ⋱  ]
    #       [       ⋱  ⋱  ]
    #
    @inbounds tridiagonalize!(M, α, β)
    #
    # compute the maximum element
    #
    #   tmax := max |Tᵢⱼ|
    #           i,j
    #
    tmax = zero(T)

    @inbounds for i in 1:n
        tmax = max(tmax, abs(α[i]))
    end

    @inbounds for i in 1:n - 1
        tmax = max(tmax, abs(β[i]))
    end

    if iszero(tmax)
        λ = zero(T)
    elseif !isfinite(tmax)
        λ = T(NaN)
    elseif isone(n)
        @inbounds λ = α[1]
    else
        #
        # rescale T by a power of two so that
        #
        #   ‖ T ‖ ≈ 1
        #
        texp = exponent(tmax)

        if !iszero(texp)
            s = ldexp(one(T), -texp)

            @inbounds for i in 1:n
                α[i] *= s
            end

            @inbounds for i in 1:n - 1
                β[i] *= s
            end
        end
        #
        # bracket the minimum eigenvalue of T:
        #
        #   lo := min Tᵢᵢ - |Tᵢᵢ₋₁| - |Tᵢ₊₁ᵢ|
        #          i
        #
        #   hi := min Tᵢᵢ
        #          i
        #
        # and compute the norm
        #
        #   nrm := ‖ T ‖∞
        #
        lo = typemax(T)
        hi = typemax(T)
        nrm = typemin(T)
        βim1 = zero(T)

        @inbounds for i in 1:n - 1
            αi   = α[i]
            βi   = β[i]

            lo = min(lo, αi - abs(βim1) - abs(βi))
            hi = min(hi, αi)
            nrm = max(nrm, abs(αi) + abs(βim1) + abs(βi))

            βim1 = βi
        end

        @inbounds αn = α[n]

        lo = min(lo, αn - abs(βim1))
        hi = min(hi, αn)
        nrm = max(nrm, abs(αn) + abs(βim1))

        mx = ldexp(λmax, -texp)

        if mx ≤ lo || (mx < hi && !checksturm(α, β, mx, n))
            λ = λmax
        else
            lo -= abs(lo) * tol
            hi  = min(hi, mx)
            #
            # compute the minimum eigenvalue of T
            # using Sturm bisection
            #
            @inbounds while hi - lo > tol * max(one(T), abs(lo), abs(hi))
                mid = (lo + hi) / 2

                if checksturm(α, β, mid, n)
                    hi = mid
                else
                    lo = mid
                end
            end

            λ = ldexp(lo, texp)
        end
    end

    return λ
end
