#
# the universal noise band of the log-form membership functions: every
# |log| of a normal T is bounded (~745 for Float64), so the absolute
# error of h near any root is bounded by a small multiple of eps(T)
#
const NBAND = 65536

# stop factor: the finder resolves the root to K·(band width); K = 8
# removes the endgame tail, K = 32 gains ~0.5 eval more but visibly
# perturbs trajectories
const NFK = 8

#
# nflog3(jet, hi) -> τ
#
# return a certified step τ ∈ [0, hi] with h(τ) ≥ 0, within NFK band
# widths of the last root of the log-form membership function h. `jet`
# returns the triple (h, h′, h″); the solver owns the noise band. The
# probes are model-selected — chord, logarithmic, or pole — against the
# observed derivatives; a wrong model costs evaluations, never
# soundness, since every accept requires h ≥ nband. A forced bisection
# guarantees one bit of contraction per iteration
#
function nflog3(jet, hi::T; tol::T = eps(T), maxit::Int = 53) where {T <: AbstractFloat}
    glo, gplo, _ = jet(zero(T))
    return nflog3(jet, hi, glo, gplo; tol = tol, maxit = maxit)
end

function nflog3(
        jet,
        hi::T,
        glo::T,
        gplo::T;
        tol::T = eps(T),
        maxit::Int = 53,
    ) where {T <: AbstractFloat}
    τb = hi
    lo = zero(T)
    nb = NBAND * eps(T)

    glo >= nb || return lo

    ghi, gphi, gpphi = jet(hi)

    ghi >= nb && return hi

    if isfinite(gphi) && !iszero(gphi)
        whi = nb / abs(gphi)
    else
        whi = zero(T)
    end

    for _ in 1:maxit
        wid = hi - lo
        #
        # lo-advancing probe: in the endgame (hi inside its own band)
        # step to the resolution limit; otherwise fit the chord, the
        # logarithmic model, and the pole model through the endpoints
        # and keep whichever best predicts the observed h′(lo)
        #
        if isfinite(ghi) && ghi >= -nb
            m = hi - 3 * max(whi, eps(T) * (one(T) + hi)) / 2
        elseif isfinite(ghi) && isfinite(glo)
            m = lo + glo * wid / (glo - ghi)

            best = abs((ghi - glo) / wid - gplo)
            dlo = τb - lo
            dhi = τb - hi

            if dhi > 0
                ulo = log(dlo)
                uhi = log(dhi)

                c = (glo - ghi) / (ulo - uhi)
                e = abs(-c / dlo - gplo)

                if e < best
                    best = e
                    m2 = τb - exp(ulo - glo * (ulo - uhi) / (glo - ghi))

                    if lo < m2 < hi
                        m = m2
                    end
                end

                b = (glo - ghi) / (inv(dhi) - inv(dlo))
                a = glo + b / dlo
                e = abs(-b / (dlo * dlo) - gplo)

                if e < best && !iszero(a)
                    m2 = τb - b / a

                    if lo < m2 < hi
                        m = m2
                    end
                end
            end
        else
            m = lo + wid / 2
        end

        if !(lo < m < hi)
            m = lo + wid / 2
        end

        g, gp, gpp = jet(m)

        if g >= nb
            lo, glo, gplo = m, g, gp
        else
            hi, ghi, gphi, gpphi = m, g, gp, gpp
            if isfinite(gp) && !iszero(gp)
                whi = nb / abs(gp)
            else
                whi = zero(T)
            end
        end

        if hi - lo < max(NFK * whi, tol * (one(T) + hi))
            break
        end
        #
        # hi-advancing probe: the ratio ρ = h″ (τb − τ) / h′ classifies
        # the boundary behavior (≈1 logarithmic, ≈2 pole, else smooth)
        # and the matched one-point model is applied; Halley, then
        # Newton, are the fallbacks
        #
        d = τb - hi
        m = T(NaN)

        if isfinite(gpphi) && isfinite(gphi) && !iszero(gphi) && d > 0
            ρ = gpphi * d / gphi

            if T(0.5) < ρ < T(1.5)
                m = τb - d * exp(ghi / (gphi * d))
            elseif T(1.5) <= ρ < T(3)
                den = ghi - gphi * d

                if !iszero(den)
                    m = τb + gphi * d * d / den
                end
            end
        end

        if !(lo < m < hi)
            m = hi - 2 * ghi * gphi / (2 * gphi * gphi - ghi * gpphi)

            if !(lo < m < hi)
                m = hi - ghi / gphi
            end
        end

        if !(lo < m < hi)
            m = lo + (hi - lo) / 2
        end

        g, gp, gpp = jet(m)

        if g >= nb
            lo, glo, gplo = m, g, gp
        else
            hi, ghi, gphi, gpphi = m, g, gp, gpp
            if isfinite(gp) && !iszero(gp)
                whi = nb / abs(gp)
            else
                whi = zero(T)
            end
        end

        if hi - lo < max(NFK * whi, tol * (one(T) + hi))
            break
        end
        #
        # guaranteed contraction: if the two probes failed to halve the
        # bracket, bisect
        #
        if 2 * (hi - lo) > wid
            m = lo + (hi - lo) / 2

            g, gp, gpp = jet(m)

            if g >= nb
                lo, glo, gplo = m, g, gp
            else
                hi, ghi, gphi, gpphi = m, g, gp, gpp
                if isfinite(gp) && !iszero(gp)
                    whi = nb / abs(gp)
                else
                    whi = zero(T)
                end
            end

            if hi - lo < max(NFK * whi, tol * (one(T) + hi))
                break
            end
        end
    end

    return lo
end
