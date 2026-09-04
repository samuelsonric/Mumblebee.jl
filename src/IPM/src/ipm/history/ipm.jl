const IPMHistoryRow{T} = @NamedTuple{μ::T, step::T, pres::T, dres::T, pobj::T, dobj::T, ρ::T, δ::T,
    piter::Int, ppass::Int, pstat::KKTStatus,
    citer::Int, cpass::Int, cstat::KKTStatus,
    dmin::T, dmax::T, χ::T}

struct IPMHistory{T} <: AbstractVector{IPMHistoryRow{T}}
    μ::Vector{T}
    step::Vector{T}
    pres::Vector{T}
    dres::Vector{T}
    pobj::Vector{T}
    dobj::Vector{T}
    ρ::Vector{T}
    δ::Vector{T}
    piter::Vector{Int}
    ppass::Vector{Int}
    pstat::Vector{KKTStatus}
    citer::Vector{Int}
    cpass::Vector{Int}
    cstat::Vector{KKTStatus}
    dmin::Vector{T}
    dmax::Vector{T}
    χ::Vector{T}
end

function IPMHistory{T}() where {T}
    return IPMHistory{T}(T[], T[], T[], T[], T[], T[], T[], T[],
        Int[], Int[], KKTStatus[],
        Int[], Int[], KKTStatus[],
        T[], T[], T[])
end

function defaultrow(::IPMHistory{T}) where {T}
    return (
        μ = T(NaN), step = zero(T), pres = T(NaN), dres = T(NaN), pobj = T(NaN), dobj = T(NaN),
        ρ = T(NaN), piter = 0, ppass = 0, pstat = KKT_SOLVED,
        citer = 0, cpass = 0, cstat = KKT_SOLVED, dmin = T(NaN), dmax = T(NaN), χ = T(NaN),
    )
end

function Base.getindex(hist::IPMHistory, i::Int)
    μ       = hist.μ[i]
    step    = hist.step[i]
    pres    = hist.pres[i]
    dres    = hist.dres[i]
    pobj    = hist.pobj[i]
    dobj    = hist.dobj[i]
    ρ       = hist.ρ[i]
    δ       = hist.δ[i]
    piter   = hist.piter[i]; ppass = hist.ppass[i]; pstat = hist.pstat[i]
    citer   = hist.citer[i]; cpass = hist.cpass[i]; cstat = hist.cstat[i]
    dmin   = hist.dmin[i]; dmax = hist.dmax[i]
    χ       = hist.χ[i]
    return (; μ, step, pres, dres, pobj, dobj, ρ, δ, piter, ppass, pstat, citer, cpass, cstat,
        dmin, dmax, χ)
end

function Base.push!(hist::IPMHistory, row::NamedTuple)
    push!(hist.μ,       row.μ)
    push!(hist.step,    row.step)
    push!(hist.pres,    row.pres)
    push!(hist.dres,    row.dres)
    push!(hist.pobj,    row.pobj)
    push!(hist.dobj,    row.dobj)
    push!(hist.ρ,       row.ρ)
    push!(hist.δ,       row.δ)
    push!(hist.piter,   row.piter); push!(hist.ppass, row.ppass); push!(hist.pstat, row.pstat)
    push!(hist.citer,   row.citer); push!(hist.cpass, row.cpass); push!(hist.cstat, row.cstat)
    push!(hist.dmin, row.dmin); push!(hist.dmax, row.dmax)
    push!(hist.χ, row.χ)
    return hist
end

function Base.empty!(hist::IPMHistory)
    empty!(hist.μ)
    empty!(hist.step)
    empty!(hist.pres)
    empty!(hist.dres)
    empty!(hist.pobj)
    empty!(hist.dobj)
    empty!(hist.ρ)
    empty!(hist.δ)
    empty!(hist.piter); empty!(hist.ppass); empty!(hist.pstat)
    empty!(hist.citer); empty!(hist.cpass); empty!(hist.cstat)
    empty!(hist.dmin); empty!(hist.dmax)
    empty!(hist.χ)
    return hist
end

function showtop(io::IO, ::IPMHistory; indent::Integer=0)
    pad = " "^indent
    println(io, pad, "┌──────┬──────────┬──────────┬───────────┬───────────┬────────┬───────┬──────────┬──────────┬──────────┬──────────┐")
    println(io, pad, "│ iter │   pres   │   dres   │   pobj    │   dobj    │  step  │ solve │    ρ     │    δ     │    μ     │    χ     │")
    return
end

function showbot(io::IO, ::IPMHistory; indent::Integer=0)
    pad = " "^indent
    println(io, pad, "└──────┴──────────┴──────────┴───────────┴───────────┴────────┴───────┴──────────┴──────────┴──────────┴──────────┘")
    return
end

function showmid(io::IO, ::IPMHistory; indent::Integer=0)
    pad = " "^indent
    println(io, pad, "├──────┼──────────┼──────────┼───────────┼───────────┼────────┼───────┼──────────┼──────────┼──────────┼──────────┤")
    println(io, pad, "│    ⋮ │        ⋮ │        ⋮ │         ⋮ │         ⋮ │      ⋮ │     ⋮ │        ⋮ │        ⋮ │        ⋮ │        ⋮ │")
    return
end

function showrow(io::IO, i::Integer, row::IPMHistoryRow; indent::Integer=0)
    pad = " "^indent
    println(io, pad, "├──────┼──────────┼──────────┼───────────┼───────────┼────────┼───────┼──────────┼──────────┼──────────┼──────────┤")
    print(io, pad)
    # solve = total triangular solve-pairs this step (predictor + corrector, each counting CG
    # iterations and refinement passes); δ = penalty, ρ = regularization.
    solve = row.piter + row.ppass + row.citer + row.cpass
    @printf(io, "│ %4d │ %8.2e │ %8.2e │ %9.2e │ %9.2e │ %6.4f │ %5d │ %8.2e │ %8.2e │ %8.2e │ %8.2e │\n",
            i, row.pres, row.dres, row.pobj, row.dobj, row.step, solve, row.ρ, row.δ, row.μ, row.χ)
    return
end

function Base.show(io::IO, ::MIME"text/plain", hist::T) where {T <: IPMHistory}
    println(io, T, ":")
    return showhistory(io, hist; indent=2)
end
