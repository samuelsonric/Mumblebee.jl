const BRAILLE_BLOCKS = (0x2801, 0x2802, 0x2804, 0x2840, 0x2808, 0x2810, 0x2820, 0x2880)

const AbstractScalar{T} = AbstractArray{T, 0}

function ispositive(i::I) where {I}
    return i > zero(I)
end

function two(::Type{I}) where {I}
    return one(I) + one(I)
end

function two(::I) where {I}
    return two(I)
end

function three(::Type{I}) where {I}
    return two(I) + one(I)
end

function three(::I) where {I}
    return three(I)
end

function four(::Type{I}) where {I}
    return two(I) + two(I)
end

function four(::I) where {I}
    return four(I)
end

function intriangle(i, j, uplo::Symbol)
    if uplo == :U
        return i <= j
    else
        return i >= j
    end
end

function intriangle(i, j, ::Val{UL}) where {UL}
    return intriangle(i, j, UL)
end

function unwrapadj(A::Transpose)
    return parent(A), Val(:T)
end

function unwrapadj(A::Adjoint)
    return parent(A), Val(:C)
end

function unwrapadj(A)
    return A, Val(:N)
end

function wrapadj(A, ::Val{TA}) where {TA}
    if TA === :N
        B =           A
    elseif TA === :T
        B = transpose(A)
    else
        B =   adjoint(A)
    end

    return B
end

function binarysearchlast(A::AbstractVector{I}, v::Integer, strt::Integer, stop::Integer) where {I <: Integer}
    return binarysearchlast(A, convert(I, v)::I, convert(I, strt)::I, convert(I, stop)::I)
end

function binarysearchlast(A::AbstractVector{I}, v::I, strt::I, stop::I) where {I <: Integer}
    lo = strt - one(I)
    hi = stop + one(I)

    while lo < hi - one(I)
        md = (lo + hi) >> 1

        if A[md] <= v
            lo = md
        else
            hi = md
        end
    end

    return lo
end

for (fname, elty) in ((:dgeqp3_, :Float64), (:sgeqp3_, :Float32))
    @eval function geqp3!(A::AbstractMatrix{$elty}, piv::AbstractVector{BlasInt}, tau::AbstractVector{$elty}, work::Vector{$elty})
        m, n = size(A)
        lda = stride(A, 2)
        info = Ref{BlasInt}()
        lwork = BlasInt(-1)

        length(work) < 1 && resize!(work, 1)

        for i in 1:2
            ccall((@blasfunc($fname), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  m, n, A, lda, piv, tau, work, lwork, info)

            chklapackerror(info[])

            if i == 1
                lwork = BlasInt(real(work[1]))
                length(work) < lwork && resize!(work, lwork)
            end
        end

        return A
    end
end

for (fname, elty, relty) in ((:dpstrf_, :Float64,    :Float64),
                             (:spstrf_, :Float32,    :Float32),
                             (:zpstrf_, :ComplexF64, :Float64),
                             (:cpstrf_, :ComplexF32, :Float32))
    @eval function pstrf!(uplo::AbstractChar, A::AbstractMatrix{$elty}, piv::AbstractVector{BlasInt}, work::AbstractVector{$relty}, tol::Real)
        n = size(A, 2)
        @assert length(piv) >= n
        @assert length(work) >= 2n

        lda = stride(A, 2)
        rank = Ref{BlasInt}()
        info = Ref{BlasInt}()

        ccall((@blasfunc($fname), libblastrampoline), Cvoid,
              (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
               Ptr{BlasInt}, Ref{$relty}, Ptr{$relty}, Ref{BlasInt}, Clong),
              uplo, n, A, lda, piv, rank, tol, work, info, 1)

        chkargsok(info[])

        return rank[]
    end
end

for (fname, elty) in ((:dgesdd_, :Float64), (:sgesdd_, :Float32))
    @eval function gesdd!(job::AbstractChar, A::AbstractMatrix{$elty}, S::AbstractVector{$elty}, U::AbstractMatrix{$elty}, V::AbstractMatrix{$elty}, work::Vector{$elty}, iwork::Vector{BlasInt})
        m, n = size(A)
        lda  = max(1, stride(A, 2))
        ldu  = max(1, stride(U, 2))
        ldv  = max(1, stride(V, 2))
        info = Ref{BlasInt}()
        lwork = BlasInt(-1)

        length(work) < 1 && resize!(work, 1)

        for i in 1:2
            ccall((@blasfunc($fname), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ref{BlasInt}, Clong),
                  job, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, iwork, info, 1)

            chklapackerror(info[])

            if i == 1
                lwork = round(BlasInt, nextfloat(real(work[1])))
                length(work) < lwork && resize!(work, lwork)
            end
        end

        return S, V
    end
end

function braille_grid(io::IO, nrow, ncol)
    maxheight, maxwidth = displaysize(io)
    maxheight -= 4
    maxwidth ÷= 2

    if get(io, :limit, true)
        scaleheight = max(8, min(nrow, 2maxwidth, 4maxheight))
        scalewidth = max(4, min(ncol, 2maxwidth, 4maxheight))
    else
        scaleheight = max(8, nrow)
        scalewidth = max(4, ncol)
    end

    rowscale = max(1, scaleheight - 1) / max(1, nrow - 1)
    colscale = max(1, scalewidth - 1) / max(1, ncol - 1)

    grow = (scalewidth - 1) ÷ 2 + 4
    gcol = (scaleheight - 1) ÷ 4 + 1

    grid = FMatrix{UInt16}(undef, grow, gcol)
    grid                 .= '⠀'
    grid[1,           :] .= '⎢'
    grid[grow - 1,    :] .= '⎥'
    grid[1,           1]  = '⎡'
    grid[1,        gcol]  = '⎣'
    grid[grow - 1,    1]  = '⎤'
    grid[grow - 1, gcol]  = '⎦'
    grid[grow,        :] .= '\n'

    return grid, rowscale, colscale
end

function setbraille!(braillegrid, row, col, rowscale, colscale)
    si = round(Int, (row - 1) * rowscale + 1)
    sj = round(Int, (col - 1) * colscale + 1)

    i =  (sj - 1) ÷ 2  + 2
    j =  (si - 1) ÷ 4  + 1
    b = ((sj - 1) % 2) * 4 + ((si - 1) % 4 + 1)

    braillegrid[i, j] |= BRAILLE_BLOCKS[b]
end
