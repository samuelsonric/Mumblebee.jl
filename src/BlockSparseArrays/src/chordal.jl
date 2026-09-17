function LinearAlgebra.axpy!(α::Number, A::BlockSparseMatrix{T}, L::ChordalTriangular{DIAG, :L, T, I}) where {DIAG, T, I}
    v = one(I)

    @inbounds for f in fronts(L)
        fD, res = diagblock(L, f)
        fL, sep = offdblock(L, f)

        rlo = first(res)
        rhi = last(res)

        if !isempty(sep)
            slo = first(sep)
            shi = last(sep)
        end

        while v ≤ nvtxs(A) && colrange(A, v) ⊆ res
            vcol = colrange(A, v)
            vlo = first(vcol)

            for e in srcrange(A, v)
                u = A.tgt[e]
                urow = rowrange(A, u)

                ulo = first(urow)
                uhi = last(urow)

                Ae = block(A, u, v, e)

                if uhi <= rhi
                    ulo < rlo && continue

                    for j in vcol
                        fj = j - rlo + one(I)
                        vj = j - vlo + one(I)

                        for i in urow
                            fi = i - rlo + one(I)
                            vi = i - ulo + one(I)
                            parent(fD)[fi, fj] += α * Ae[vi, vj]
                        end
                    end
                elseif !isempty(sep) && ulo >= slo && uhi <= shi
                    k = one(I)

                    while sep[k] < ulo
                        k += one(I)
                    end

                    for j in vcol
                        fj = j - rlo + one(I)
                        vj = j - vlo + one(I)

                        kk = k

                        for i in urow
                            vi = i - ulo + one(I)
                            fL[kk, fj] += α * Ae[vi, vj]
                            kk += one(I)
                        end
                    end
                end
            end

            v += one(I)
        end
    end

    return L
end

function LinearAlgebra.axpby!(α::Number, A::BlockSparseMatrix, β::Number, L::ChordalTriangular)
    if iszero(β)
        fill!(L, β)
    elseif !isone(β)
        rmul!(L, β)
    end

    axpy!(α, A, L)
    return L
end


function LinearAlgebra.axpy!(α::Number, A::BlockSparseMatrix{T}, L::ChordalTriangular{DIAG, :U, T, I}) where {DIAG, T, I}
    vr = one(I)
    vs = one(I)

    @inbounds for f in fronts(L)
        fD, res = diagblock(L, f)
        fL, sep = offdblock(L, f)

        rlo = first(res)
        rhi = last(res)

        while vr ≤ nvtxs(A) && colrange(A, vr) ⊆ res
            vcol = colrange(A, vr)
            vlo = first(vcol)

            for e in srcrange(A, vr)
                u = A.tgt[e]
                urow = rowrange(A, u)

                ulo = first(urow)
                uhi = last(urow)

                Ae = block(A, u, vr, e)

                ulo < rlo && continue
                uhi > rhi && continue

                for j in vcol
                    fj = j - rlo + one(I)
                    vj = j - vlo + one(I)

                    for i in urow
                        fi = i - rlo + one(I)
                        vi = i - ulo + one(I)
                        parent(fD)[fi, fj] += α * Ae[vi, vj]
                    end
                end
            end

            vr += one(I)
        end

        isempty(sep) && continue

        while vs ≤ nvtxs(A) && last(colrange(A, vs)) < first(sep)
            vs += one(I)
        end

        while vs ≤ nvtxs(A) && colrange(A, vs) ⊆ sep
            vcol = colrange(A, vs)
            vlo = first(vcol)

            k = one(I)

            while sep[k] < vlo
                k += one(I)
            end

            for e in srcrange(A, vs)
                u = A.tgt[e]
                urow = rowrange(A, u)

                ulo = first(urow)
                uhi = last(urow)

                Ae = block(A, u, vs, e)

                ulo < rlo && continue
                uhi > rhi && continue

                for j in vcol
                    fj = k + (j - vlo)
                    vj = j - vlo + one(I)

                    for i in urow
                        fi = i - rlo + one(I)
                        vi = i - ulo + one(I)
                        fL[fi, fj] += α * Ae[vi, vj]
                    end
                end
            end

            vs += one(I)
        end
    end

    return L
end

function Base.copyto!(L::ChordalTriangular, A::BlockSparseMatrix)
    axpby!(true, A, false, L)
    return L
end
