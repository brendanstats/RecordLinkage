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
    out = BitArray(na, nb, nmeasure)
    trueC = MatchMatrix(Array{Int64}(0), Array{Int64}(0), na, nb)
    for ii in 1:na
        for jj in 1:nb
            if recordsA[ii] == recordsB[jj]
                add_match!(trueC, ii, jj)
                out[ii, jj, :] = rand(nmeasure) .< pM
            else
                out[ii, jj, :] = rand(nmeasure) .< pU
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
    out = BitArray(C.nrow, C.ncol, nmeasure)
    for ii in eachindex(out[:, :, 1])
        out[ii, :] = rand(nmeasure) .< pU
    end
    for kk in 1:length(C.rows)
        out[C.rows[kk], C.cols[kk], :] = rand(nmeasure) .< pM
    end
    return out
end

"""
Simulate record linkage based on true population size and two samples drawn from it.  errorRate is rate at which errors are made in observations, in which case the observation is re-drawn from levels.  Returned values pM and pU indicate the probability of observations matching given an underlying match (pM) or no underlying match (pU)
"""
function single_linkage_levels{T <: Integer, G <: AbstractFloat}(nRecords::T, nA::T, nB::T, levels::Array{T, 1}, errorRate::Array{G, 1}; blocking::Bool = false, blocks::Array{T, 1} = Array{T}(0))
    if nA > nRecords
        error("Number of records must be a least as large as number of records in group A")
    end
    if nB > nRecords
        error("Number of records must be a least as large as number of records in group B")
    end
    if any(errorRate .> 1.0) || any(errorRate .< 0.0)
        error("errorRate must be between 0 and 1")
    end
    if length(errorRate) != length(levels)
        error("if vector supplied for error rate length must match length of levels")
    end
    
    #Generate underlying population data
    nLevels = length(levels)
    dataRecords = Array{T}(nRecords, nLevels)
    for jj in  1:nLevels
        dataRecords[:, jj] = StatsBase.sample(1:levels[jj], nRecords, replace = true)
    end
    
    #Sample from Population
    recordsA = StatsBase.sample(1:nRecords, nA, replace = false)
    recordsB = StatsBase.sample(1:nRecords, nB, replace = false)
    
    #Generate data for recordsA with errors
    dataA = Array{T}(nA, nLevels)
    for ii in 1:nA
        for jj in 1:nLevels
            if rand() < errorRate[jj]
                dataA[ii, jj] = StatsBase.sample(1:levels[jj])
            else
                dataA[ii, jj] = dataRecords[recordsA[ii], jj]
            end
        end
    end

    #Generate data for recordsB with errors
    dataB = Array{T}(nB, nLevels)
    for ii in 1:nB
        for jj in 1:nLevels
            if rand() < errorRate[jj]
                dataB[ii, jj] = StatsBase.sample(1:levels[jj])
            else
                dataB[ii, jj] = dataRecords[recordsB[ii], jj]
            end
        end
    end

    if blocking
        dataA = dataA[sortperm(dataA[:, 1]), :]
        dataB = dataB[sortperm(dataB[:, 1]), :]
    end
    
    #Determine Matches
    out = BitArray(nA, nB, nLevels)
    trueC = MatchMatrix(Array{Int64}(0), Array{Int64}(0), nA, nB)
    for ii in 1:nA
        for jj in 1:nB
            if recordsA[ii] == recordsB[jj]
                    add_match!(trueC, ii, jj)
            end
            out[ii, jj, :] = dataA[ii, :] .== dataB[jj, :]
        end
    end
    pM = Array{G}(nLevels)
    pU = Array{G}(nLevels)
    for (ii, (k, ε)) in enumerate(zip(levels, errorRate))
        #pM[ii] = ((k - 1) * ε^2 + (2 - k) * ε) / k + 1.0
        pM[ii] = (1.0 - ε)^2 + (1.0 / k) * ε * (2.0 - ε)
        pU[ii] = 1 / k
    end
    return out, trueC, pM, pU
end

single_linkage_levels{T <: Integer, G <: AbstractFloat}(nRecords::T, nA::T, nB::T, levels::Array{T, 1}, errorRate::G) = single_linkage_levels(nRecords, nA, nB, levels, fill(errorRate, length(levels)))

single_linkage_levels{T <: Integer, G <: AbstractFloat}(nRecords::T, nA::T, nB::T, levels::Array{T, 1}, errorRate::G, blocking::Bool) = single_linkage_levels(nRecords, nA, nB, levels, fill(errorRate, length(levels)), blocking = blocking)

"""
Estimate MatchMatrix by identifying all elements in the data that have at least as many matches as specified, the default is the dimension of the observation.  Then a random permuation of the indicies is chosen.  Indicies are looped through and kept if both the row and column do not yet have a link in them.
"""
function estimate_C0{G <: Integer}(data::BitArray{3}, threshold::G = size(data, 3))
    matchcounts = sum(data, 3)
    n, m = size(matchcounts)
    rowopen = trues(n)
    colopen = trues(m)
    idxs = find(matchcounts) do x
        x >= threshold
    end
    rows = Array{Int64}()
    cols = Array{Int64}()
    for idx in idxs[randperm(length(idxs))]
        ii, jj = ind2sub((n, m), idx)
        if rowopen[ii] && colopen[jj]
            rowopen[ii] = false
            colopen[jj] = false
            push!(rows, ii)
            push!(cols, jj)
    end
    return MatchMatrix{Int64}(rows, cols, n, m)
end
