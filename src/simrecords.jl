#import StatsBase
#include("matching_matrix.jl")

"""
Simulate data corresponding to binary comparison metrics for record linkage, enforcing single linkage of records

Requires with a total number of records as well as the number of records sampled in each of the two populations compared
along with M and U probability vectors.  In this case an underlying MatchMatrix will be returned along with the simulated
comparison values.  Comparisons are returned in the form of a 3-dimensional Array of 0s and 1s

Alternatively a MatchMatrix can be supplied along with M and U probability vectors.  In this case only the simulated data
will be returned.  

* Examples
simulate_singlelinkage_binary(1000, 300, 200, [0.8, 0.82, 0.95, 0.98], [0.2, 0.3, 0.38, 0.41])
C = MatchMatrix(StatsBase.sample(1:300, 15, replace = false), StatsBase.sample(1:200, 15, replace = false), 300, 200)
simulate_singlelinkage_binary(C, [0.8, 0.82, 0.95, 0.98], [0.2, 0.3, 0.38, 0.41])
"""
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
    #Sample from Population
    recordsA = StatsBase.sample(1:nrecords, na, replace = false)
    recordsB = StatsBase.sample(1:nrecords, nb, replace = false)
    
    #Determine Matches
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

"""
Add a simulation method consistent with blocking
"""
function simulate_singlelinkage_binary{G <: AbstractFloat, T <: Integer}(nrecords::T, na::T, nb::T, obpop::Array{T, 1}, pM::Array{G, 1}, pU::Array{G, 1})
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
    nmeasure = length(pM)
    popdata = Array{T}(nrecords, nmeasure)
    for ii in 1:nmeasure
        popdata[:, ii] = StatsBase.sample(1:obpop[ii], nrecords, replace = true)
    end
    #Sample from Population
    recordsA = StatsBase.sample(1:nrecords, na, replace = false)
    recordsB = StatsBase.sample(1:nrecords, nb, replace = false)
    
    #Determine Matches
    nmeasure = length(pM)
    out = BitArray(nmeasure, na, nb)
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


function simulate_singlelinkage_binary{G <: AbstractFloat}(C::MatchMatrix, pM::Array{G, 1}, pU::Array{G, 1})
    if any(pM .< 0.0) || any(pM .> 1.0)
        error("M probabilities must be between 0 and 1")
    end
    if any(pU .< 0.0) || any(pU .> 1.0)
        error("U probabilities must be between 0 and 1")
    end
    if length(pM) != length(pU)
        error("Length of M probabilities and U probabilities must match")
    end
    nmeasure = length(pM)
    out = Array{Int8}(nmeasure, C.nrow, C.ncol)
    for ii in eachindex(out[1, :, :])
        out[:, ii] = Int8.(rand(nmeasure) .< pU)
    end
    for kk in 1:length(C.rows)
        out[:, C.rows[kk], C.cols[kk]] = Int8.(rand(nmeasure) .< pM)
    end
    return out
end

"""
Convert a 2D Array into a single column and add columns indicating row and column index in original array.  Order is row, col, value
"""
function gridtoarray{G <: Real}(x::Array{G, 2})
    nrow, ncol = size(x)
    return [repeat(1:nrow, outer=ncol) repeat(1:ncol, inner=nrow) vec(x)]
end
