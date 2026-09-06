const IPMHistoryRow{T} = @NamedTuple{pres::T, dres::T, pobj::T, dobj::T, step::T,
    piter::Int, ppass::Int, pstat::KKTStatus,
    citer::Int, cpass::Int, cstat::KKTStatus,
    dmin::T, dmax::T, ρ::T, δ::T, μ::T, χ::T}

struct IPMHistory{T} <: AbstractVector{IPMHistoryRow{T}}
    pres::Vector{T}
    dres::Vector{T}
    pobj::Vector{T}
    dobj::Vector{T}
    step::Vector{T}
    piter::Vector{Int}
    ppass::Vector{Int}
    pstat::Vector{KKTStatus}
    citer::Vector{Int}
    cpass::Vector{Int}
    cstat::Vector{KKTStatus}
    dmin::Vector{T}
    dmax::Vector{T}
    ρ::Vector{T}
    δ::Vector{T}
    μ::Vector{T}
    χ::Vector{T}
end

function IPMHistory{T}() where {T}
    return IPMHistory{T}(T[], T[], T[], T[], T[],
        Int[], Int[], KKTStatus[],
        Int[], Int[], KKTStatus[],
        T[], T[], T[], T[], T[], T[])
end

function defaultrow(::IPMHistory{T}) where {T}
    return (
        pres = T(NaN), dres = T(NaN), pobj = T(NaN), dobj = T(NaN), step = zero(T),
        piter = 0, ppass = 0, pstat = KKT_SOLVED,
        citer = 0, cpass = 0, cstat = KKT_SOLVED,
        dmin = T(NaN), dmax = T(NaN), ρ = T(NaN), δ = zero(T), μ = T(NaN), χ = T(NaN),
    )
end

function Base.getindex(hist::IPMHistory, i::Int)
    pres    = hist.pres[i]
    dres    = hist.dres[i]
    pobj    = hist.pobj[i]
    dobj    = hist.dobj[i]
    step    = hist.step[i]
    piter   = hist.piter[i]; ppass = hist.ppass[i]; pstat = hist.pstat[i]
    citer   = hist.citer[i]; cpass = hist.cpass[i]; cstat = hist.cstat[i]
    dmin    = hist.dmin[i]; dmax = hist.dmax[i]
    ρ       = hist.ρ[i]
    δ       = hist.δ[i]
    μ       = hist.μ[i]
    χ       = hist.χ[i]
    return (; pres, dres, pobj, dobj, step, piter, ppass, pstat, citer, cpass, cstat,
        dmin, dmax, ρ, δ, μ, χ)
end

function Base.push!(hist::IPMHistory, row::NamedTuple)
    push!(hist.pres,    row.pres)
    push!(hist.dres,    row.dres)
    push!(hist.pobj,    row.pobj)
    push!(hist.dobj,    row.dobj)
    push!(hist.step,    row.step)
    push!(hist.piter,   row.piter); push!(hist.ppass, row.ppass); push!(hist.pstat, row.pstat)
    push!(hist.citer,   row.citer); push!(hist.cpass, row.cpass); push!(hist.cstat, row.cstat)
    push!(hist.dmin, row.dmin); push!(hist.dmax, row.dmax)
    push!(hist.ρ, row.ρ); push!(hist.δ, row.δ); push!(hist.μ, row.μ)
    push!(hist.χ, row.χ)
    return hist
end

function Base.empty!(hist::IPMHistory)
    empty!(hist.pres)
    empty!(hist.dres)
    empty!(hist.pobj)
    empty!(hist.dobj)
    empty!(hist.step)
    empty!(hist.piter); empty!(hist.ppass); empty!(hist.pstat)
    empty!(hist.citer); empty!(hist.cpass); empty!(hist.cstat)
    empty!(hist.dmin); empty!(hist.dmax)
    empty!(hist.ρ); empty!(hist.δ); empty!(hist.μ)
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
