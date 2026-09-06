"""
    CofreeCone <: AbstractCone

The cone of all n-dimensional Euclidean
vectors.
"""
struct CofreeCone <: AbstractCone end

struct CofreeConeCache <: AbstractCache{CofreeCone}
    cone::CofreeCone
end

function CofreeConeCache()
    return CofreeConeCache(CofreeCone())
end

############################################################################################
# degree
############################################################################################

function degree(::CofreeCone, n::Integer)
    return 0
end

############################################################################################
# cachesize
############################################################################################

function cachesize(::Type{CofreeCone}, n::Integer)
    return 0
end

############################################################################################
# cache
############################################################################################

function cache(::Caches, ::Integer, c::CofreeCone)
    return CofreeConeCache(c)
end

############################################################################################
# identity!
############################################################################################

function identity!(x::AbstractVector{T}, ::CofreeCone) where {T}
    fill!(x, zero(T))
    return x
end

############################################################################################
# scale!
############################################################################################

function scale!(::AbstractMatrix{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::CofreeConeCache, ::ConeWorkspace) where {T}
    return true, zero(T)
end

function scale!(::CofreeCone, ::Integer, ::BlockSparseMatrix{T}, ::BlockSparseMatrix, ::Caches, ::AbstractVector, ::AbstractVector, ::BlockSparseMatrix, ::ConeWorkspace) where {T}
    return true, zero(T)
end

############################################################################################
# corr!
############################################################################################

function corr!(
        r::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        ::AbstractVector{T},
        ::Real,
        ::CofreeConeCache,
        ::ConeWorkspace,
    ) where {T}
    fill!(r, zero(T))
    return r
end

############################################################################################
# corr0!
############################################################################################

function corr0!(r::AbstractVector{T}, ::AbstractVector, ::AbstractVector, ::Real, ::CofreeConeCache, ::ConeWorkspace) where {T}
    fill!(r, zero(T))
    return r
end

############################################################################################
# maxsteps
############################################################################################

function maxsteps(::AbstractVector{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::CofreeConeCache, ::ConeWorkspace) where {T}
    return one(T), one(T)
end

############################################################################################
# dualshadow! / primalshadow!
############################################################################################

function dualshadow!(sd::AbstractVector{T}, ::AbstractVector{T}, ::CofreeConeCache, ::ConeWorkspace) where {T}
    fill!(sd, zero(T))
    return true
end

function primalshadow!(sp::AbstractVector{T}, ::AbstractVector{T}, ::CofreeConeCache, ::ConeWorkspace) where {T}
    fill!(sp, zero(T))
    return true
end

############################################################################################
# primalhess!
############################################################################################

function primalhess!(r::AbstractVector, p::AbstractVector, Δp::AbstractVector, ::CofreeConeCache, work::ConeWorkspace)
    return fill!(r, false)
end

############################################################################################
# primalthird!
############################################################################################

function primalthird!(r::AbstractVector, p::AbstractVector, Δp1::AbstractVector, Δp2::AbstractVector, ::CofreeConeCache, work::ConeWorkspace)
    return fill!(r, false)
end
