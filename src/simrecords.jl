import StatsBase
include("matching_matrix.jl")

function simulate_singlelinkage_binary{G <: AbstractFloat, T <: Integer}(nrecords::T, na::T, nb::T, pM::Array{G, 1}, pU::Array{G, 1})
    if na > nrecords
        error("Number of records must be a least as large as number of records in group A")
    end
    if nb > nrecords
        error("Number of records must be a least as large as number of records in group B")
    end
    if any(pM .< 0.0) || any(pM .> 1.0)
        error("M probabilities must be between 0 and 1")
    end
    if any(pU .< 0.0) || any(pU .> 1.0)
        error("U probabilities must be between 0 and 1")
    end
    if length(pM) != length(pU)
        error("Length of M probabilities and U probabilities must match")
    end

    recordsA = StatsBase.sample(1:nrecords, na, replace = false)
    recordsB = StatsBase.sample(1:nrecords, nb, replace = false)
    nmeasure = length(pM)
    out = Array{Int64}(nmeasure, na, nb)
    trueC = MatchMatrix(Array{Int64}(0), Array{Int64}(0), na, nb)
    for ii in 1:na
        for jj in 1:nb
            if recordsA[ii] == recordsB[jj]
                add_match!(trueC, ii, jj)
                out[:, ii, jj] = Int64.(rand(nmeasure) .< pM)
            else
                out[:, ii, jj] = Int64.(rand(nmeasure) .< pU)
            end
        end
    end
    return out, trueC
end
