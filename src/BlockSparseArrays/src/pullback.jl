function pullback(A::BlockSparseMatrix{T, I}, G::SparseMatrixCSC{<:Any, I}, U::AbstractVector{I}, V::AbstractVector{I}) where {T, I}
    nBout = convert(I, size(G, 1))
    nBvtx = convert(I, size(G, 2))
    nBarc = convert(I, nnz(G))
    Bxsrc = G.colptr
    Btgt  = G.rowval

    return pullback(A, nBout, nBvtx, nBarc, Bxsrc, Btgt, U, V)
end

function pullback(A::BlockSparseMatrix{T, I}, nBout::I, nBvtx::I, nBarc::I, Bxsrc::AbstractVector{I}, Btgt::AbstractVector{I}, Qmap::AbstractVector{I}, Pmap::AbstractVector{I}) where {T, I}
    Bxrow, plus, mnus = pull_scan(A, nBout, nBvtx, nBarc, Bxsrc, Btgt, Qmap, Pmap)

    B = pull_init_B(A, nBout, nBvtx, nBarc, Bxsrc, Btgt, Bxrow, Pmap)

    pull_symb_B!(B, A, Pmap)

    pull_numb_B!(B, A, plus, mnus, Qmap, Pmap)

    return B
end

function pull_scan(A::BlockSparseMatrix{T, I}, nBout::I, nBvtx::I, nBarc::I, Bxsrc::AbstractVector{I}, Btgt::AbstractVector{I}, Qmap::AbstractVector{I}, Pmap::AbstractVector{I}) where {T, I}
    nAvtx = nvtxs(A)

    # ======================================================================

    Pxsrc = FVector{I}(undef, nAvtx + one(I))
    Ptgt  = FVector{I}(undef, nBvtx)

    push_prim!(Pxsrc, Ptgt, Pmap, nBvtx, nAvtx, Val(:U))

    # ======================================================================

    Bxrow = FVector{I}(undef, nBout + one(I))
    mark  = FVector{I}(undef, nBout)
    prev  = FVector{I}(undef, nBout)

    plus = FVector{I}(undef, nBarc)
    mnus = FVector{I}(undef, nBarc)

    for Bu in oneto(nBout)
        Au = Qmap[Bu]

        if Au > zero(I)
            Bxrow[Bu + one(I)] = nrows(A, Au)
        else
            Bxrow[Bu + one(I)] = zero(I)
        end

        mark[Bu] = zero(I)
    end

    for Be in oneto(nBarc)
        plus[Be] = -one(I)
        mnus[Be] = -one(I)
    end

    # ======================================================================

    for Av in oneto(nAvtx)
        vAcol = ncols(A, Av)

        Pebgn = Pxsrc[Av]
        Peend = Pxsrc[Av + one(I)] - one(I)

        for Pe in Pebgn:Peend
            Bv = Ptgt[Pe]

            Bebgn = Bxsrc[Bv]
            Beend = Bxsrc[Bv + one(I)] - one(I)

            for Be in Bebgn:Beend
                Bu = Btgt[Be]

                if mark[Bu] == Av
                    uBbgn = Bxrow[Bu + one(I)]

                    plus[prev[Bu]] = uBbgn
                    mnus[Be]       = uBbgn

                    Bxrow[Bu + one(I)] = uBbgn + vAcol
                else
                    mark[Bu] = Av
                end

                prev[Bu] = Be
            end
        end
    end

    # ======================================================================

    Bxrow[one(I)] = Bi = one(I)

    for Bu in oneto(nBout)
        Bxrow[Bu + one(I)] = Bi += Bxrow[Bu + one(I)]
    end

    return Bxrow, plus, mnus
end

function pull_init_B(A::BlockSparseMatrix{T, I}, nBout::I, nBvtx::I, nBarc::I, Bxsrc::AbstractVector{I}, Btgt::AbstractVector{I}, Bxrow::AbstractVector{I}, Pmap::AbstractVector{I}) where {T, I}
    @assert nBout >= zero(I)
    @assert nBvtx >= zero(I)
    @assert nBarc >= zero(I)
    @assert length(Bxrow) > nBout
    @assert length(Pmap) >= nBvtx

    # ======================================================================

    nBcol = zero(I)

    for Bv in oneto(nBvtx)
        nBcol += ncols(A, Pmap[Bv])
    end

    # ======================================================================

    nBrow = Bxrow[nBout + one(I)] - one(I)

    # ======================================================================

    nBbnz = zero(I)

    for Bv in oneto(nBvtx)
        vBcol = ncols(A, Pmap[Bv])

        Bebgn = Bxsrc[Bv]
        Beend = Bxsrc[Bv + one(I)] - one(I)

        for Be in Bebgn:Beend
            Bu = Btgt[Be]
            nBbnz += (Bxrow[Bu + one(I)] - Bxrow[Bu]) * vBcol
        end
    end

    # ======================================================================

    Bxcol = FVector{I}(undef, nBvtx + one(I))
    Bxblk = FVector{I}(undef, nBarc + one(I))
    Bval  = FVector{T}(undef, nBbnz)

    return BlockSparseMatrix{T, I}(nBout, nBvtx, nBarc, nBcol, nBrow, nBbnz, Bxsrc, Bxcol, Bxrow, Bxblk, Btgt, Bval)
end

function pull_symb_B!(B::BlockSparseMatrix{T, I}, A::BlockSparseMatrix{T, I}, Pmap::AbstractVector{I}) where {T, I}
    @assert length(Pmap) >= nvtxs(B)

    nBvtx = nvtxs(B)
    nBarc = narcs(B)

    # ======================================================================

    B.xcol[one(I)] = Bj = one(I)

    for Bv in oneto(nBvtx)
        Bj += ncols(A, Pmap[Bv])
        B.xcol[Bv + one(I)] = Bj
    end

    # ======================================================================

    Bb = one(I)

    for Bv in oneto(nBvtx)
        vBcol = B.xcol[Bv + one(I)] - B.xcol[Bv]

        for Be in srcrange(B, Bv)
            Bu = B.tgt[Be]
            uBrow = B.xrow[Bu + one(I)] - B.xrow[Bu]

            B.xblk[Be] = Bb
            Bb += uBrow * vBcol
        end
    end

    B.xblk[nBarc + one(I)] = Bb

    return B
end

function pull_numb_B!(B::BlockSparseMatrix{T, I}, A::BlockSparseMatrix{T, I}, plus::AbstractVector{I}, mnus::AbstractVector{I}, Qmap::AbstractVector{I}, Pmap::AbstractVector{I}) where {T, I}
    @assert length(Qmap) >= nouts(B)
    @assert length(Pmap) >= nvtxs(B)
    @assert length(plus) >= narcs(B)
    @assert length(mnus) >= narcs(B)

    nBvtx = nvtxs(B)

    # ======================================================================

    mark = FVector{I}(undef, nouts(A))

    for Bv in oneto(nBvtx)
        Av = Pmap[Bv]

        for Ae in srcrange(A, Av)
            mark[A.tgt[Ae]] = Ae
        end

        vAcol = ncols(A, Av)

        for Be in srcrange(B, Bv)
            Bu = B.tgt[Be]
            Au = Qmap[Bu]

            Pibgn = plus[Be]
            Mibgn = mnus[Be]

            Bx = block(B, Bu, Bv, Be)

            fill!(Bx, zero(T))

            if Au > zero(I) && Mibgn < zero(I)
                Ae = mark[Au]; uArow = nrows(A, Au)

                By = view(Bx, oneto(uArow), :)
                Ax = block(A, Au, Av, Ae)

                copyto!(By, Ax)
            end

            if Pibgn >= zero(I)
                for Sj in oneto(vAcol)
                    Bx[Pibgn + Sj, Sj] = one(T)
                end
            end

            if Mibgn >= zero(I)
                for Sj in oneto(vAcol)
                    Bx[Mibgn + Sj, Sj] = -one(T)
                end
            end
        end
    end

    return B
end
