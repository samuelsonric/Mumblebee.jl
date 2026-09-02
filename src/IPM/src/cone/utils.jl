# ============================================================================
# cross - 3-vector cross product
# ============================================================================

@propagate_inbounds function crossstatic!(z::AbstractVector, x::AbstractVector, y::AbstractVector)
    @boundscheck checkbounds(z, 3)
    @boundscheck checkbounds(x, 3)
    @boundscheck checkbounds(y, 3)

    @inbounds z[1] = muladd(x[2], y[3], -x[3] * y[2])
    @inbounds z[2] = muladd(x[3], y[1], -x[1] * y[3])
    @inbounds z[3] = muladd(x[1], y[2], -x[2] * y[1])

    return z
end

# ============================================================================
# dot - vector inner product
# ============================================================================

@propagate_inbounds function dotstatic(x::AbstractVector, y::AbstractVector, ::Val{N}) where {N}
    @boundscheck checkbounds(x, N)
    @boundscheck checkbounds(y, N)

    s = zero(promote_eltype(x, y))

    @inbounds for i in 1:N
        s = muladd(x[i], y[i], s)
    end

    return s
end

# ============================================================================
# norm - vector Euclidean norm
# ============================================================================

@propagate_inbounds function normstatic(x::AbstractVector, ::Val{N}) where {N}
    @boundscheck checkbounds(x, N)

    s = zero(eltype(x))

    @inbounds for i in 1:N
        s = muladd(x[i], x[i], s)
    end

    return sqrt(s)
end

# ============================================================================
# copy - element-by-element copy
# ============================================================================

@propagate_inbounds function copystatic!(y::AbstractVector, x::AbstractVector, ::Val{N}) where {N}
    @boundscheck checkbounds(y, N)
    @boundscheck checkbounds(x, N)

    @inbounds for i in 1:N
        y[i] = x[i]
    end

    return y
end

for n in 1:8
    body = [:(A[$i,$j] = B[$i,$j]) for j in 1:n for i in 1:n]

    @eval @propagate_inbounds function copystatic!(A::AbstractMatrix, B::AbstractMatrix, ::Val{$n})
        @boundscheck checkbounds(A, $n, $n)
        @boundscheck checkbounds(B, $n, $n)

        @inbounds begin
            $(body...)
        end

        return A
    end
end

# ============================================================================
# axpy - y = a*x + y
# ============================================================================

@propagate_inbounds function axpystatic!(a, x::AbstractVector, y::AbstractVector, ::Val{N}) where {N}
    @boundscheck checkbounds(x, N)
    @boundscheck checkbounds(y, N)

    @inbounds for i in 1:N
        y[i] = muladd(a, x[i], y[i])
    end

    return y
end

# ============================================================================
# axpby - y = a*x + b*y
# ============================================================================

@propagate_inbounds function axpbystatic!(a, x::AbstractVector, b, y::AbstractVector, ::Val{N}) where {N}
    @boundscheck checkbounds(x, N)
    @boundscheck checkbounds(y, N)

    @inbounds for i in 1:N
        y[i] = muladd(a, x[i], b * y[i])
    end

    return y
end

# ============================================================================
# lmul - left multiply in place, by a scalar or a triangular matrix
# ============================================================================

@propagate_inbounds function lmulstatic!(a::Number, x::AbstractVector, ::Val{N}) where {N}
    @boundscheck checkbounds(x, N)

    @inbounds for i in 1:N
        x[i] *= a
    end

    return x
end

@propagate_inbounds function lmulstatic!(a::Number, M::AbstractMatrix, ::Val{N}) where {N}
    @boundscheck checkbounds(M, N, N)

    @inbounds for j in 1:N, i in 1:N
        M[i, j] *= a
    end

    return M
end

for n in 1:8, (Rtype, upper) in ((:UpperTriangular, true), (:LowerTriangular, false))
    body = Expr[]

    if upper
        rows = 1:n
    else
        rows = n:-1:1
    end

    for c in 1:n, i in rows
        if upper
            prange = i:n
        else
            prange = 1:i
        end

        acc = foldr((p, a) -> :(muladd(RP[$i,$p], A[$p,$c], $a)), prange; init = :(zero(T)))
        push!(body, :(A[$i,$c] = $acc))
    end

    @eval @propagate_inbounds function lmulstatic!(R::$Rtype, A::AbstractMatrix{T}, ::Val{$n}) where {T}
        RP = parent(R)

        @boundscheck checkbounds(RP, $n, $n)
        @boundscheck checkbounds(A, $n, $n)

        @inbounds begin
            $(body...)
        end

        return A
    end
end

# ============================================================================
# ldiv - left division in place, by a scalar or a triangular matrix
# ============================================================================

@propagate_inbounds function ldivstatic!(a::Number, x::AbstractVector, ::Val{N}) where {N}
    @boundscheck checkbounds(x, N)

    @inbounds for i in 1:N
        x[i] /= a
    end

    return x
end

for n in 1:8, (Rtype, lower) in ((:LowerTriangular, true), (:UpperTriangular, false))
    body = Expr[]

    if lower
        rows = 1:n
    else
        rows = n:-1:1
    end

    for c in 1:n, r in rows
        num = :(M[$r,$c])

        if lower
            prange = 1:r-1
        else
            prange = r+1:n
        end

        for p in prange
            num = :(muladd(-RP[$r,$p], M[$p,$c], $num))
        end

        push!(body, :(M[$r,$c] = ($num) / RP[$r,$r]))
    end

    @eval @propagate_inbounds function ldivstatic!(R::$Rtype, M::AbstractMatrix, ::Val{$n})
        RP = parent(R)

        @boundscheck checkbounds(RP, $n, $n)
        @boundscheck checkbounds(M, $n, $n)

        @inbounds begin
            $(body...)
        end

        return M
    end
end

# ============================================================================
# mul - matrix multiplication
# ============================================================================

function mulstatic!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::Val{N}) where {N}
    return mulstatic!(C, A, B, true, false, Val(N))
end

@propagate_inbounds function mulstatic!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α::Number, β::Number, ::Val{N}) where {N}
    @boundscheck checkbounds(C, N, N)
    @boundscheck checkbounds(A, N, N)
    @boundscheck checkbounds(B, N, N)

    T = promote_eltype(A, B)

    @inbounds for j in 1:N, i in 1:N
        s = zero(T)

        for k in 1:N
            s = muladd(A[i, k], B[k, j], s)
        end

        if iszero(β)
            C[i, j] = α * s
        else
            C[i, j] = muladd(β, C[i, j], α * s)
        end
    end

    return C
end

@propagate_inbounds function mulstatic!(C::AbstractVector, A::AbstractMatrix, B::AbstractVector, α::Number, β::Number, ::Val{N}) where {N}
    @boundscheck checkbounds(C, N)
    @boundscheck checkbounds(A, N, N)
    @boundscheck checkbounds(B, N)

    T = promote_eltype(A, B)

    if iszero(β)
        @inbounds for i in 1:N
            s = zero(T)

            for k in 1:N
                s = muladd(A[i, k], B[k], s)
            end

            C[i] = α * s
        end
    else
        @inbounds for i in 1:N
            s = zero(T)

            for k in 1:N
                s = muladd(A[i, k], B[k], s)
            end

            C[i] = muladd(β, C[i], α * s)
        end
    end

    return C
end

# ============================================================================
# ger - rank-1 update: M = α*x*y' + β*M
# ============================================================================

@propagate_inbounds function gerstatic!(M::AbstractMatrix, x::AbstractVector, y::AbstractVector, α::Number, β::Number, ::Val{N}) where {N}
    @boundscheck checkbounds(M, N, N)
    @boundscheck checkbounds(x, N)
    @boundscheck checkbounds(y, N)

    if iszero(β)
        @inbounds for j in 1:N
            yj = y[j]

            for i in 1:N
                M[i, j] = α * x[i] * yj
            end
        end
    else
        @inbounds for j in 1:N
            yj = y[j]

            for i in 1:N
                M[i, j] = muladd(β, M[i, j], α * x[i] * yj)
            end
        end
    end

    return M
end

# ============================================================================
# syrk - symmetric rank-k update
# ============================================================================

function syrk!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, α::T, β::T) where {T <: BlasFloat}
    return LinearAlgebra.BLAS.syrk!('L', 'N', α, A, β, C)
end

function syrk!(C::AbstractMatrix{T}, A::AdjOrTrans{T, <:AbstractMatrix{T}}, α::T, β::T) where {T <: BlasFloat}
    return LinearAlgebra.BLAS.syrk!('L', 'T', α, parent(A), β, C)
end

function syrk!(C::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    return syrk!(C, A, one(T), zero(T))
end

for n in 1:8
    zbody = Expr[]
    ebody = Expr[]

    for j in 1:n, i in j:n
        acc = foldr((k, a) -> :(muladd(A[$i, $k], A[$j, $k], $a)), 1:n; init = :(zero(T)))
        push!(zbody, :(C[$i, $j] = α * $acc))
        push!(ebody, :(C[$i, $j] = muladd(β, C[$i, $j], α * $acc)))
    end

    @eval @propagate_inbounds function syrkstatic!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, α::Number, β::Number, ::Val{$n}) where {T}
        @boundscheck checkbounds(C, $n, $n)
        @boundscheck checkbounds(A, $n, $n)

        @inbounds begin
            if iszero(β)
                $(zbody...)
            else
                $(ebody...)
            end
        end

        return C
    end
end

function syrkstatic!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, ::Val{N}) where {T, N}
    return syrkstatic!(C, A, one(T), zero(T), Val(N))
end

# ============================================================================
# chol - Cholesky factorization (inlined, in-place, returns success)
# ============================================================================

for n in 1:8
    body = Expr[]

    for j in 1:n
        djj = :(A[$j,$j])

        for p in 1:j-1
            djj = :(muladd(-A[$j,$p], A[$j,$p], $djj))
        end

        push!(body, :(djj = $djj))
        push!(body, :(djj > zero(T) || return false))
        push!(body, :(ljj = sqrt(djj)))
        push!(body, :(A[$j,$j] = ljj))

        for i in j+1:n
            num = :(A[$i,$j])

            for p in 1:j-1
                num = :(muladd(-A[$i,$p], A[$j,$p], $num))
            end

            push!(body, :(A[$i,$j] = ($num) / ljj))
        end
    end

    @eval @propagate_inbounds function cholstatic!(M::Symmetric{T}, ::Val{$n}) where T
        A = parent(M)

        @boundscheck checkbounds(A, $n, $n)

        @inbounds begin
            $(body...)
        end

        return true
    end
end

# ============================================================================
# rdiv - right triangular solves (inlined)
# ============================================================================

for n in 1:8, (Rtype, upper) in ((:UpperTriangular, true), (:LowerTriangular, false))
    body = Expr[]

    if upper
        cols = 1:n
    else
        cols = n:-1:1
    end

    for c in cols, i in 1:n
        num = :(M[$i,$c])

        if upper
            prange = 1:c-1
        else
            prange = c+1:n
        end

        for p in prange
            num = :(muladd(-M[$i,$p], RP[$p,$c], $num))
        end

        push!(body, :(M[$i,$c] = ($num) / RP[$c,$c]))
    end

    @eval @propagate_inbounds function rdivstatic!(M::AbstractMatrix, R::$Rtype, ::Val{$n})
        RP = parent(R)

        @boundscheck checkbounds(RP, $n, $n)
        @boundscheck checkbounds(M, $n, $n)

        @inbounds begin
            $(body...)
        end

        return M
    end
end

# ============================================================================
# symm - fill the strict upper triangle from the lower (symmetric M)
# ============================================================================

function symmetrize!(M::AbstractMatrix)
    for j in axes(M, 1)
        for i in 1:j - 1
            M[i, j] = M[j, i]
        end
    end

    return M
end

@propagate_inbounds function symmstatic!(M::AbstractMatrix, ::Val{N}) where {N}
    @boundscheck checkbounds(M, N, N)

    @inbounds for j in 1:N, i in 1:j-1
        M[i, j] = M[j, i]
    end

    return M
end

# ============================================================================
# skron - symmetric Kronecker product H += A ⊗ A (svec), inlined
# ============================================================================

function skron!(H::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    α = roottwo(T)
    tll = 1

    @inbounds for l in 1:n
        tij = 0

        for j in 1:n
            Ajl = A[j, l]

            tij += 1; H[tij, tll] += Ajl^2

            for i in j + 1:n
                tij += 1; H[tij, tll] += α * A[i, l] * Ajl
            end
        end

        tkl = tll

        for k in l + 1:n
            tkl += 1; tij = 0

            for j in 1:n
                Ajk = A[j, k]
                Ajl = A[j, l]

                tij += 1; H[tij, tkl] += α * Ajk * Ajl

                for i in j + 1:n
                    tij += 1; H[tij, tkl] += A[i, k] * Ajl + A[i, l] * Ajk
                end
            end
        end

        tll += n - l + 1
    end

    return H
end

for n in 1:8
    body = Expr[:(α = roottwo(T))]

    tll = 1

    for l in 1:n
        tij = 0

        for j in 1:n
            tij += 1
            push!(body, :(H[$tij, $tll] += A[$j,$l]^2))

            for i in j+1:n
                tij += 1
                push!(body, :(H[$tij, $tll] += α * A[$i,$l] * A[$j,$l]))
            end
        end

        tkl = tll

        for k in l+1:n
            tkl += 1
            tij = 0

            for j in 1:n
                tij += 1
                push!(body, :(H[$tij, $tkl] += α * A[$j,$k] * A[$j,$l]))

                for i in j+1:n
                    tij += 1
                    push!(body, :(H[$tij, $tkl] += A[$i,$k] * A[$j,$l] + A[$i,$l] * A[$j,$k]))
                end
            end
        end

        tll += n - l + 1
    end

    @eval @propagate_inbounds function skronstatic!(H::AbstractMatrix{T}, A::AbstractMatrix{T}, ::Val{$n}) where {T}
        @boundscheck checkbounds(A, $n, $n)
        @boundscheck checkbounds(H, $(n*(n+1)÷2), $(n*(n+1)÷2))

        @inbounds begin
            $(body...)
        end

        return H
    end
end

# ============================================================================
# svec - symmetric matrix → svec vector
# ============================================================================

@propagate_inbounds function svec!(v::AbstractVector{T}, M::AbstractMatrix{T}) where {T}
    n = size(M, 1)
    @boundscheck checkbounds(M, n, n)
    @boundscheck checkbounds(v, n * (n + 1) ÷ 2)

    k = 0
    α = roottwo(T)

    @inbounds for j in 1:n
        k += 1; v[k] = M[j, j]

        for i in j + 1:n
            k += 1; v[k] = α * M[i, j]
        end
    end

    return v
end

for n in 1:8
    body = Expr[:(α = roottwo(T))]

    k = 0

    for j in 1:n
        k += 1
        push!(body, :(v[$k] = M[$j,$j]))

        for i in j+1:n
            k += 1
            push!(body, :(v[$k] = α * M[$i,$j]))
        end
    end

    @eval @propagate_inbounds function svecstatic!(v::AbstractVector{T}, M::AbstractMatrix{T}, ::Val{$n}) where {T}
        @boundscheck checkbounds(M, $n, $n)
        @boundscheck checkbounds(v, $(n * (n + 1) ÷ 2))

        @inbounds begin
            $(body...)
        end

        return v
    end
end

# ============================================================================
# smat - svec → symmetric matrix (lower triangle)
# ============================================================================

@propagate_inbounds function smat!(M::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    n = size(M, 1)
    @boundscheck checkbounds(M, n, n)
    @boundscheck checkbounds(v, n * (n + 1) ÷ 2)

    k = 0
    α = roottwo(T)

    @inbounds for j in 1:n
        k += 1; M[j, j] = v[k]

        for i in j + 1:n
            k += 1; M[i, j] = v[k] / α
        end
    end

    return M
end

for n in 1:8
    body = Expr[:(α = roottwo(T))]

    k = 0

    for j in 1:n
        k += 1
        push!(body, :(M[$j,$j] = v[$k]))

        for i in j+1:n
            k += 1
            push!(body, :(M[$i,$j] = v[$k] / α))
        end
    end

    @eval @propagate_inbounds function smatstatic!(M::AbstractMatrix{T}, v::AbstractVector, ::Val{$n}) where {T}
        @boundscheck checkbounds(M, $n, $n)
        @boundscheck checkbounds(v, $(n * (n + 1) ÷ 2))

        @inbounds begin
            $(body...)
        end

        return M
    end
end
