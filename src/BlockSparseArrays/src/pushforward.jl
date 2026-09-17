# ==========================================================================
# pushforward
# ==========================================================================

function pushforward(A::BlockSparseMatrix{T, I}, G::SparseMatrixCSC{<:Any, I}, U::AbstractVector{I}, V::AbstractVector{I}; kw...) where {T, I}
    nout = convert(I, size(G, 1))
    nvtx = convert(I, size(G, 2))
    narc = convert(I, nnz(G))
    xsrc = G.colptr
    tgt  = G.rowval

    return pushforward(A, nout, nvtx, narc, xsrc, tgt, U, V; kw...)
end

function pushforward(
        A::BlockSparseMatrix{T, I},
        nout::I,
        nvtx::I,
        narc::I,
        xsrc::AbstractVector{I},
        tgt::AbstractVector{I},
        U::AbstractVector{I},
        V::AbstractVector{I};
        atol::Real = 0,
        rtol::Real = atol > 0 ? zero(T) : eps(T),
        qr::Bool = false,
    ) where {T, I}

    return pushforward(A, nout, nvtx, narc, xsrc, tgt, U, V, convert(T, atol), convert(T, rtol), qr)
end

function pushforward(
        A::BlockSparseMatrix{T, I},
        nCout::I,
        nCvtx::I,
        nCarc::I,
        Cxsrc::AbstractVector{I},
        Ctgt::AbstractVector{I},
        Qmap::AbstractVector{I},
        Pmap::AbstractVector{I},
        atol::T,
        rtol::T,
        qr::Bool,
    ) where {T, I}

    @assert nCout >= zero(I)
    @assert nCvtx >= zero(I)
    @assert nCarc >= zero(I)
    @assert length(Cxsrc) > nCvtx
    @assert length(Ctgt) >= nCarc
    @assert length(Qmap) >= nouts(A)
    @assert length(Pmap) >= nvtxs(A)
    @assert atol >= 0
    @assert rtol >= 0

    nAout = nouts(A)

    Qxsrc = FVector{I}(undef, nCvtx + one(I))
    Qtgt  = FVector{I}(undef, nAout)

    Rxsrc = FVector{I}(undef, nCout + one(I))
    Rtgt  = FVector{I}(undef, nAout)

    B = push_make_B!(Qxsrc, Qtgt, Rtgt, A, nCvtx, Qmap, Pmap, atol, rtol, qr)
    C = push_make_C!(Rxsrc, Rtgt, Qtgt, Qxsrc, A, B, nCout, nCarc, Cxsrc, Ctgt, Qmap)

    return B, C
end

# ==========================================================================
# push_make_B
# ==========================================================================

function push_null_chol!(M::AbstractMatrix, N::AbstractMatrix, piv::AbstractVector, work::AbstractVector, atol::Real, rtol::Real)
    @assert size(N, 1) >= size(M, 2)
    @assert size(N, 2) >= size(M, 2)
    @assert length(piv) >= size(M, 2)
    @assert length(work) >= 2size(M, 2)
    @assert atol >= 0
    @assert rtol >= 0

    m, n = size(M)

    if n == 0
        k = 0
    elseif m == 0
        #
        #   N ← I
        #
        for j in 1:n
            for i in 1:n
                N[i, j] = 0
            end

            N[j, j] = 1
        end

        k = n
    else
        #
        # form the Gram matrix:
        #
        #   N = Mᵀ M
        #
        syrk!('U', 'T', true, M, false, N)
        #
        # compute the pivot tolerance
        #
        #   ϵ = max(atol, rtol n max Nᵢᵢ)
        #                         i
        #
        λ = zero(eltype(N))

        for i in 1:n
            λ = max(λ, N[i, i])
        end

        tol = max(atol, rtol * n * λ)
        #
        # factorize N:
        #
        #   Pᵀ N P = Rᵀ R
        #
        r = pstrf!('U', N, piv, work, tol)
        k = n - r
        #
        # divide the upper triangular factor R
        # into blocks
        #
        #          r   k
        #   R = [ Rrr Rrk ] r
        #       [     Rkk ] k
        #
        # and solve
        #
        #   Rrk ← Rrr⁻¹ Rrk
        #
        if r > 0
            Nrr = view(N, 1:r,     1:r)
            Nrk = view(N, 1:r, r + 1:n)
            ldiv!(UpperTriangular(Nrr), Nrk)
        end
        #
        # write the nullspace to the first
        # k columns of N
        #
        #             k
        #   Pᵀ N = [ -Rrk ] r
        #          [  I   ] k
        #
        for j in 1:k
            for i in 1:n
                N[i, j] = 0
            end

            for i in 1:r
                N[piv[i], j] = -N[i, r + j]
            end

            N[piv[r + j], j] = 1
        end
    end

    return k
end

function push_null_qr!(M::AbstractMatrix, N::AbstractMatrix, piv::AbstractVector, tau::AbstractVector, work::Vector, atol::Real, rtol::Real)
    @assert size(N, 1) >= size(M, 2)
    @assert size(N, 2) >= size(M, 2)
    @assert length(piv) >= size(M, 2)
    @assert length(tau) >= min(size(M, 1), size(M, 2))
    @assert atol >= 0
    @assert rtol >= 0

    m, n = size(M); mn = min(m, n)

    if n == 0
        k = 0
    elseif m == 0
        #
        #   N ← I
        #
        for j in 1:n
            for i in 1:n
                N[i, j] = 0
            end

            N[j, j] = 1
        end

        k = n
    else
        for j in 1:n
            piv[j] = 0
        end
        #
        # factorize M:
        #
        #   M Pᵀ = Q R
        #
        geqp3!(M, piv, tau, work)
        #
        # compute the pivot tolerance
        #
        #   ϵ = max(atol, rtol n max Rᵢᵢ²)
        #                         i
        #
        tol = max(atol, rtol * n * M[1, 1]^2)
        #
        # compute the rank r and nullity k:
        #
        #   r := rank(M)
        #   k := null(M)
        #
        r = 0

        for i in 1:mn
            M[i, i]^2 > tol || break
            r += 1
        end

        k = n - r
        #
        # divide the upper triangular factor R
        # into blocks
        #
        #          r   k
        #   R = [ Rrr Rrk ] r
        #       [     Rkk ] k
        #
        # and solve
        #
        #   Rrk ← Rrr⁻¹ Rrk
        #
        if r > 0
            Mrr = view(M, 1:r,     1:r)
            Mrk = view(M, 1:r, r + 1:n)
            ldiv!(UpperTriangular(Mrr), Mrk)
        end
        #
        # write the nullspace to the first
        # k columns of N
        #
        #             k
        #   Pᵀ N = [ -Rrk ] r
        #          [  I   ] k
        #
        for j in 1:k
            for i in 1:n
                N[i, j] = 0
            end

            for i in 1:r
                N[piv[i], j] = -M[i, r + j]
            end

            N[piv[r + j], j] = 1
        end
    end

    return k
end

function push_prim!(
        xsrc::AbstractVector{I},
        tgt::AbstractVector{I},
        src::AbstractVector{I},
        nout::I,
        nvtx::I,
        ::Val{SIGN},
    ) where {I, SIGN}
    @assert nvtx >= zero(I)
    @assert length(xsrc) > nvtx
    @assert length(tgt) >= nout
    @assert length(src) >= nout
    @assert SIGN === :P || SIGN === :N || SIGN === :U

    for w in oneto(nvtx + one(I))
        xsrc[w] = zero(I)
    end

    for u in oneto(nout)
        if SIGN === :N
            v = -src[u]
        else
            v =  src[u]
        end

        if SIGN === :U || v > zero(I)
            if v < nvtx
                xsrc[v + two(I)] += one(I)
            end
        end
    end

    xsrc[one(I)] = e = one(I)

    for v in oneto(nvtx)
        xsrc[v + one(I)] = e += xsrc[v + one(I)]
    end

    for u in oneto(nout)
        if SIGN === :N
            v = -src[u]
        else
            v =  src[u]
        end

        if SIGN === :U || v > zero(I)
            tgt[xsrc[v + one(I)]] = u
            xsrc[v + one(I)] += one(I)
        end
    end

    return xsrc, tgt
end

function push_make_B!(
        Qxsrc::AbstractVector{I},
        Qtgt::AbstractVector{I},
        Mxrow::AbstractVector{I},
        A::BlockSparseMatrix{T, I},
        nBvtx::I,
        Qmap::AbstractVector{I},
        Pmap::AbstractVector{I},
        atol::T,
        rtol::T,
        qr::Bool,
    ) where {T, I}
    @assert nBvtx >= zero(I)
    @assert length(Qxsrc) > nBvtx
    @assert length(Qtgt) >= nouts(A)
    @assert length(Mxrow) >= nouts(A)
    @assert length(Qmap) >= nouts(A)
    @assert length(Pmap) >= nvtxs(A)
    @assert atol >= 0
    @assert rtol >= 0

    nAout = nouts(A)
    nAvtx = nvtxs(A)
    nAarc = narcs(A)
    nAcol = ncols(A)
    nArow = nrows(A)
    nAbnz = nbnzs(A)

    nBout = nAvtx
    nBarc = nBout
    nBrow = nAcol

    # ======================================================================

    Bxsrc = FVector{I}(undef, nBvtx + one(I))
    Btgt  = FVector{I}(undef, nBout)

    push_prim!(Bxsrc, Btgt, Pmap, nAvtx, nBvtx, Val(:U))

    # ======================================================================

    push_prim!(Qxsrc, Qtgt, Qmap, nAout, nBvtx, Val(:N))

    # ======================================================================

    mMbnz = zero(I)
    mNbnz = zero(I)
    mMcol = zero(I)

    for Bv in oneto(nBvtx)
        Qebgn = Qxsrc[Bv]
        Qeend = Qxsrc[Bv + one(I)] - one(I)
        nMrow = zero(I)

        for Qe in Qebgn:Qeend
            nMrow += nrows(A, Qtgt[Qe])
        end

        Pebgn = Bxsrc[Bv]
        Peend = Bxsrc[Bv + one(I)] - one(I)
        nMcol = zero(I)

        for Pe in Pebgn:Peend
            nMcol += ncols(A, Btgt[Pe])
        end

        mMbnz = max(mMbnz, nMrow * nMcol)
        mNbnz = max(mNbnz, nMcol * nMcol)
        mMcol = max(mMcol, nMcol)
    end

    # ======================================================================

    Bxcol = FVector{I}(undef, nBvtx + one(I))
    Bxrow = FVector{I}(undef, nBout + one(I))
    Bxblk = FVector{I}(undef, nBarc + one(I))

    copyto!(Bxrow, 1, A.xcol, 1, nBout + one(I))

    # ======================================================================

    Bval  = T[]

    Mval  = FVector{T}(undef, mMbnz)
    Nval  = FVector{T}(undef, mNbnz)
    piv = Vector{BlasInt}(undef, mMcol)

    if qr
        work = T[]
        tau  = Vector{T}(undef, mMcol)
    else
        work = Vector{T}(undef, 2mMcol)
    end

    nBcol = Bb = zero(I)

    for Bv in oneto(nBvtx)
        Bxcol[Bv] = nBcol + one(I)

        Pebgn = Bxsrc[Bv]
        Peend = Bxsrc[Bv + one(I)] - one(I)

        Qebgn = Qxsrc[Bv]
        Qeend = Qxsrc[Bv + one(I)] - one(I)

        nMcol = zero(I)

        for Pe in Pebgn:Peend
            nMcol += ncols(A, Btgt[Pe])
        end

        nMrow = zero(I)

        for Qe in Qebgn:Qeend
            Au = Qtgt[Qe]
            Mxrow[Au] = nMrow + one(I)
            nMrow += nrows(A, Au)
        end

        M = reshape(view(Mval, oneto(nMrow * nMcol)), nMrow, nMcol)
        N = reshape(view(Nval, oneto(nMcol * nMcol)), nMcol, nMcol)

        fill!(M, zero(T)); Mj = one(I)

        for Pe in Pebgn:Peend
            Av = Btgt[Pe]; vAcol = ncols(A, Av)

            for Ae in srcrange(A, Av)
                Au = A.tgt[Ae]; uArow = nrows(A, Au)

                if Qmap[Au] < zero(I)
                    Mi = Mxrow[Au]

                    Mx = view(M, Mi:Mi + uArow - one(I), Mj:Mj + vAcol - one(I))
                    Ax = block(A, Au, Av, Ae)

                    copyto!(Mx, Ax)
                end
            end

            Mj += vAcol
        end

        if qr
            nNcol = convert(I, push_null_qr!(M, N, piv, tau, work, atol, rtol))
        else
            nNcol = convert(I, push_null_chol!(M, N, piv, work, atol, rtol))
        end

        Nibgn = one(I)

        for Pe in Pebgn:Peend
            Av = Btgt[Pe]; vAcol = ncols(A, Av)

            Bxblk[Pe] = Bb + one(I)

            Niend = Nibgn + vAcol - one(I)

            for Nj in oneto(nNcol)
                for Ni in Nibgn:Niend
                    push!(Bval, N[Ni, Nj])
                end
            end

            Bb += nNcol * vAcol

            Nibgn = Niend + one(I)
        end

        nBcol += nNcol
    end

    nBbnz = Bb
    Bxcol[nBvtx + one(I)] = nBcol + one(I)
    Bxblk[nBarc + one(I)] = nBbnz + one(I)

    return BlockSparseMatrix{T, I}(nBout, nBvtx, nBarc, nBcol, nBrow, nBbnz, Bxsrc, Bxcol, Bxrow, Bxblk, Btgt, Bval)
end

# ==========================================================================
# push_make_C
# ==========================================================================

function push_make_C!(
        Rxsrc::AbstractVector{I},
        Rtgt::AbstractVector{I},
        Stgt::AbstractVector{I},
        flag::AbstractVector{I},
        A::BlockSparseMatrix{T, I},
        B::BlockSparseMatrix{T, I},
        nCout::I,
        nCarc::I,
        Cxsrc::AbstractVector{I},
        Ctgt::AbstractVector{I},
        uhm::AbstractVector{I},
    ) where {T, I}
    @assert nCout >= zero(I)
    @assert nCarc >= zero(I)
    @assert length(Rxsrc) > nCout
    @assert length(Rtgt) >= nouts(A)
    @assert length(Stgt) >= nouts(A)
    @assert length(flag) >= nvtxs(B)
    @assert length(Cxsrc) > nvtxs(B)
    @assert length(Ctgt) >= nCarc
    @assert length(uhm) >= nouts(A)

    nAout = nouts(A)
    nCvtx = nvtxs(B)

    push_prim!(Rxsrc, Rtgt, uhm, nAout, nCout, Val(:P))

    C = push_init_C(A, B, nCout, nCvtx, nCarc, Cxsrc, Ctgt, Rxsrc, Rtgt)

    push_symb_C!(C, A, B, Rxsrc, Rtgt)

    push_numb_C!(Stgt, flag, C, A, B, uhm, Rxsrc, Rtgt)

    return C
end

function push_init_C(A::BlockSparseMatrix{T, I}, B::BlockSparseMatrix{T, I}, nCout::I, nCvtx::I, nCarc::I, Cxsrc::AbstractVector{I}, Ctgt::AbstractVector{I}, Rxsrc::AbstractVector{I}, Rtgt::AbstractVector{I}) where {T, I}
    @assert nCout >= zero(I)
    @assert nCvtx >= zero(I)
    @assert nCarc >= zero(I)
    @assert nvtxs(B) == nCvtx
    @assert length(Cxsrc) > nCvtx
    @assert length(Ctgt) >= nCarc
    @assert length(Rxsrc) > nCout

    nCcol = ncols(B)

    # ======================================================================

    nCrow = zero(I)

    for Cu in oneto(nCout)
        Rebgn = Rxsrc[Cu]
        Reend = Rxsrc[Cu + one(I)] - one(I)

        for Re in Rebgn:Reend
            nCrow += nrows(A, Rtgt[Re])
        end
    end

    # ======================================================================

    nCbnz = zero(I)

    for Cv in oneto(nCvtx)
        vCcol = ncols(B, Cv)

        Cebgn = Cxsrc[Cv]
        Ceend = Cxsrc[Cv + one(I)] - one(I)

        for Ce in Cebgn:Ceend
            Cu = Ctgt[Ce]

            Rebgn = Rxsrc[Cu]
            Reend = Rxsrc[Cu + one(I)] - one(I)
            uCrow = zero(I)

            for Re in Rebgn:Reend
                uCrow += nrows(A, Rtgt[Re])
            end

            nCbnz += uCrow * vCcol
        end
    end

    # ======================================================================

    Cxrow = FVector{I}(undef, nCout + one(I))
    Cxblk = FVector{I}(undef, nCarc + one(I))
    Cval  = FVector{T}(undef, nCbnz)

    return BlockSparseMatrix{T, I}(nCout, nCvtx, nCarc, nCcol, nCrow, nCbnz, Cxsrc, B.xcol, Cxrow, Cxblk, Ctgt, Cval)
end

function push_symb_C!(C::BlockSparseMatrix{T, I}, A::BlockSparseMatrix{T, I}, B::BlockSparseMatrix{T, I}, Rxsrc::AbstractVector{I}, Rtgt::AbstractVector{I}) where {T, I}
    @assert nvtxs(C) == nvtxs(B)
    @assert length(Rxsrc) > nouts(C)

    nCout = nouts(C)
    nCvtx = nvtxs(C)
    nCarc = narcs(C)

    # ======================================================================

    C.xrow[one(I)] = Ci = one(I)

    for Cu in oneto(nCout)
        Rebgn = Rxsrc[Cu]
        Reend = Rxsrc[Cu + one(I)] - one(I)

        for Re in Rebgn:Reend
            Ci += nrows(A, Rtgt[Re])
        end

        C.xrow[Cu + one(I)] = Ci
    end

    # ======================================================================

    Cb = one(I)

    for Cv in oneto(nCvtx)
        vCcol = ncols(B, Cv)

        for Ce in srcrange(C, Cv)
            Cu = C.tgt[Ce]
            uCrow = C.xrow[Cu + one(I)] - C.xrow[Cu]

            C.xblk[Ce] = Cb
            Cb += uCrow * vCcol
        end
    end

    C.xblk[nCarc + one(I)] = Cb

    return C
end

function push_gemm!(C::BlockSparseMatrix{T, I}, A::BlockSparseMatrix{T, I}, B::BlockSparseMatrix{T, I}, Rmap::AbstractVector{I}, mark::AbstractVector{I}, Stgt::AbstractVector{I}, Cv::Integer) where {T, I}
    @assert one(I) <= Cv <= nvtxs(C)
    @assert length(Rmap) >= nouts(A)
    @assert length(mark) >= nouts(C)
    @assert length(Stgt) >= nouts(A)

    for Be in srcrange(B, Cv)
        Av = B.tgt[Be]
        Bx = block(B, Av, Cv, Be)

        for Ae in srcrange(A, Av)
            Au = A.tgt[Ae]; uArow = nrows(A, Au)

            if Rmap[Au] > zero(I)
                Cu = Rmap[Au]
                Ce = mark[Cu]

                Ax = block(A, Au, Av, Ae)
                Cx = block(C, Cu, Cv, Ce)

                Sibgn = Stgt[Au]
                Siend = Sibgn + uArow - one(I)
                Cy = view(Cx, Sibgn:Siend, :)

                mul!(Cy, Ax, Bx, one(T), one(T))
            end
        end
    end

    return C
end

function push_axpy!(C::BlockSparseMatrix{T, I}, A::BlockSparseMatrix{T, I}, B::BlockSparseMatrix{T, I}, Rmap::AbstractVector{I}, mark::AbstractVector{I}, Stgt::AbstractVector{I}, Cv::Integer) where {T, I}
    @assert one(I) <= Cv <= nvtxs(C)
    @assert length(Rmap) >= nouts(A)
    @assert length(mark) >= nouts(C)
    @assert length(Stgt) >= nouts(A)

    Sjbgn = one(I)

    for Be in srcrange(B, Cv)
        Av = B.tgt[Be]; vAcol = ncols(A, Av)

        Sjend = Sjbgn + vAcol - one(I)

        for Ae in srcrange(A, Av)
            Au = A.tgt[Ae]; uArow = nrows(A, Au)

            if Rmap[Au] > zero(I)
                Cu = Rmap[Au]
                Ce = mark[Cu]

                Ax = block(A, Au, Av, Ae)
                Cx = block(C, Cu, Cv, Ce)

                Sibgn = Stgt[Au]
                Siend = Sibgn + uArow - one(I)
                Cy = view(Cx, Sibgn:Siend, Sjbgn:Sjend)

                axpy!(one(T), Ax, Cy)
            end
        end

        Sjbgn = Sjend + one(I)
    end

    return C
end

function push_numb_C!(
        Stgt::AbstractVector{I},
        flag::AbstractVector{I},
        C::BlockSparseMatrix{T, I},
        A::BlockSparseMatrix{T, I},
        B::BlockSparseMatrix{T, I},
        Rmap::AbstractVector{I},
        Rxsrc::AbstractVector{I},
        Rtgt::AbstractVector{I},
    ) where {T, I}
    @assert nvtxs(C) == nvtxs(B)
    @assert length(Stgt) >= nouts(A)
    @assert length(flag) >= nvtxs(C)
    @assert length(Rmap) >= nouts(A)
    @assert length(Rxsrc) > nouts(C)

    nCout = nouts(C)
    nCvtx = nvtxs(C)
    nAout = nouts(A)

    # ======================================================================

    for Cu in oneto(nCout)
        Rebgn = Rxsrc[Cu]
        Reend = Rxsrc[Cu + one(I)] - one(I)
        Si = one(I)

        for Re in Rebgn:Reend
            Au = Rtgt[Re]
            Stgt[Au] = Si
            Si += nrows(A, Au)
        end
    end

    # ======================================================================

    for Cv in oneto(nCvtx)
        flag[Cv] = one(I)
    end

    for Au in oneto(nAout)
        Cv = Rmap[Au]

        if Cv < zero(I)
            flag[-Cv] = zero(I)
        end
    end

    # ======================================================================

    mark = Rxsrc

    fill!(C, zero(T))

    for Cv in oneto(nCvtx)
        for Ce in srcrange(C, Cv)
            Cu = C.tgt[Ce]
            mark[Cu] = Ce
        end

        if flag[Cv] != zero(I)
            push_axpy!(C, A, B, Rmap, mark, Stgt, Cv)
        else
            push_gemm!(C, A, B, Rmap, mark, Stgt, Cv)
        end
    end

    return C
end
