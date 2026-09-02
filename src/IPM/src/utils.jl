function tounion(v::AbstractVector, perm::AbstractVector=eachindex(v))
    return tounion(mapreduce(typeof, tmerge, v; init=Union{}), v, perm)
end

function tounion(::Type{T}, v::AbstractVector, perm::AbstractVector=eachindex(v)) where {T}
    w = FVector{T}(undef, length(v))

    for i in eachindex(w)
        w[i] = v[perm[i]]
    end

    return w
end

function two(::Type{T}) where {T}
    return convert(T, 2)
end

function signfloor(x::T, ε::T) where T
    if abs(x) < ε
        return copysign(ε, x)
    else
        return x
    end
end

# twosum: s + e = a + b exactly, where s = fl(a+b)
function twosum(a::T, b::T) where {T}
    s  = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)
end

# twoprod: p + e = a * b exactly, where p = fl(a*b)  (requires fma)
function twoprod(a::T, b::T) where {T}
    p = a * b
    return p, fma(a, b, -p)
end

# cdot: dot(p,d) = Σ p_i d_i, compensated to ~2u
function cdot(p::AbstractVector{T}, d::AbstractVector{T}) where {T}
    @assert length(p) == length(d)
    n = length(p)
    s = c = zero(T)

    @inbounds for i in 1:n
        pr, e = twoprod(p[i], d[i])
        s, e2 = twosum(s, pr)
        c += e + e2
    end

    return s + c
end

@propagate_inbounds function cdotstatic(p::AbstractVector, d::AbstractVector, ::Val{N}) where {N}
    @boundscheck checkbounds(p, N)
    @boundscheck checkbounds(d, N)

    s = c = zero(promote_eltype(p, d))

    @inbounds for i in 1:N
        pr, e = twoprod(p[i], d[i])
        s, e2 = twosum(s, pr)
        c += e + e2
    end

    return s + c
end

function twolog(x::Float64)
    t = log(Double64(x))
    return HI(t), LO(t)
end

function twoexp(ah::Float64, al::Float64)
    t = exp(Double64(ah, al))
    return HI(t), LO(t)
end

function twolog1p(ah::Float64, al::Float64)
    t = log1p(Double64(ah, al))
    return HI(t), LO(t)
end

@inline function twodiv(ah, al, bh, bl)
    q1 = ah / bh
    r  = muladd(-q1, bh, ah) + al - q1 * bl
    return twosum(q1, r / bh)
end

function weightedmean(a, b, x, y)
    return (a * x + b * y) / (a + b)
end

function weightedgraph(B::BlockSparseMatrix{T, I}, Q::BlockSparseMatrix{T, I}) where {T, I}
    weight = FVector{I}(undef, nvtxs(B))

    for v in vtxs(B)
        weight[v] = ncols(B, v)
    end

    BG = BipartiteGraph(nouts(B), nvtxs(B), narcs(B), B.xsrc, B.tgt)
    QG = BipartiteGraph(nouts(Q), nvtxs(Q), narcs(Q), Q.xsrc, Q.tgt)
    return weight, uniongraph(QG, linegraph(BG))
end

function allocblockdiag(A::BlockSparseMatrix{T, I}) where {T, I}
    nout = nvtx = narc = nvtxs(A)
    ncol = nrow = ncols(A)

    nbnz = zero(I)

    for v in vtxs(A)
        nbnz += ncols(A, v)^2
    end

    D = BlockSparseMatrix{T, I}(nout, nvtx, narc, ncol, nrow, nbnz)
    return allocblockdiag!(D, A)
end

function allocblockdiag!(D::BlockSparseMatrix{T, I}, A::BlockSparseMatrix{T, I}) where {T, I}
    nout = nvtx = narc = nvtxs(D)
    ncol = nrow = ncols(D)
    nbnz = nbnzs(D)

    blk = zero(I)

    for v in vtxs(A)
        D.xsrc[v] = v
        D.xcol[v] = A.xcol[v]
        D.xrow[v] = A.xcol[v]
        D.xblk[v] = blk + one(I)
        D.tgt[v]  = v

        blk += ncols(A, v)^2
    end

    D.xsrc[nvtx + one(I)] = nout + one(I)
    D.xcol[nvtx + one(I)] = ncol + one(I)
    D.xrow[nout + one(I)] = nrow + one(I)
    D.xblk[narc + one(I)] = nbnz + one(I)

    return D
end


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
