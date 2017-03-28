#=
Suite of Functions to compute quantities related to the number of single linkage matrices given matrix dimensions both given a number of linkages and 

code_llvm()
=#

"""
Return the log of the number of single linkage structures possible for an nrow by ncol linkage matrix with n links
"""
function logcount_single_linkage{G <: Integer}(nlink::G, nrow::G, ncol::G)
    if nlink > nrow || nlink > ncol
        error("No possible single linkage structures")
    elseif nlink == 0
        return 0.0
    end
    out = logfactorial(nrow) + logfactorial(ncol)
    out -= logfactorial(nlink)
    out -= logfactorial(nrow - nlink)
    out -= logfactorial(ncol - nlink)    
    return out
end

"""
Return the log of the number of single linkage structures possible for an nrow by ncol linkage matrix, calculation done on long scale
"""
function logcount_single_linkage{G <: Integer}(nrow::G, ncol::G)
    if nrow < 1 || ncol < 1
        error("nrow and ncol must both be postive integers")
    end
    maxLinks = min(nrow, ncol)
    linkRatios = Array{Float64}(maxLinks)
    for ii in 1:maxLinks
        linkRatios[ii] = logratio_single_linkage(ii - 1, nrow, ncol)
    end
    logCounts = cumsum(linkRatios)
    maxLogCounts = maximum(logCounts)
    out = exp(-maxLogCounts)
    for ct in logCounts
        out += exp(ct - maxLogCounts)
    end
    return log(out) + maxLogCounts
end

"""
Log number of all single linkage matricies with `nrow` rows and `ncol` columns by number of linkages, divided by maximum number of structures for specific number of linkages.
"""
function logproportion_single_linkage{G <: Integer}(nrow::G, ncol::G)
    if nrow < 1 || ncol < 1
        error("nrow and ncol must both be postive integers")
    end
    maxLinks = min(nrow, ncol)
    linkRatios = Array{Float64}(maxLinks)
    for ii in 1:maxLinks
        linkRatios[ii] = logratio_single_linkage(ii - 1, nrow, ncol)
    end
    logCounts = cumsum(linkRatios)
    maxLogCounts = Base.maximum(logCounts)
    out = Array{eltype(maxLogCounts)}(maxLinks + 1)
    out[1] = -maxLogCounts
    for (ii, lct) in enumerate(logCounts)
        out[ii + 1] = lct - maxLogCounts
    end
    return out
end

"""
penalty scales linearly with number of linkages
"""
function logproportion_single_linkage{G <: Integer, T <: AbstractFloat}(nrow::G, ncol::G, logpenalty::T)
    if nrow < 1 || ncol < 1
        error("nrow and ncol must both be postive integers")
    end
    maxLinks = min(nrow, ncol)
    linkRatios = Array{Float64}(maxLinks)
    for ii in 1:maxLinks
        linkRatios[ii] = logratio_single_linkage(ii - 1, nrow, ncol, logpenalty)
    end
    logCounts = cumsum(linkRatios)
    maxLogCounts = Base.maximum(logCounts)
    out = Array{eltype(maxLogCounts)}(maxLinks + 1)
    out[1] = -maxLogCounts
    for (ii, lct) in enumerate(logCounts)
        out[ii + 1] = lct - maxLogCounts
    end
    return out
end

"""
Log percentage of all single linkage matricies with `nrow` rows and `ncol` columns by number of linkages.
"""
function logprobability_single_linkage{G <: Integer}(nrow::G, ncol::G)
    out = logproportion_single_linkage(nrow, ncol)
    return out .- logsum(out)
end

"""
penalty scales linearly with number of linkages
"""
function logprobability_single_linkage{G <: Integer, T <: AbstractFloat}(nrow::G, ncol::G, logpenalty::T)
    out = logproportion_single_linkage(nrow, ncol, logpenalty)
    return out .- logsum(out)
end


"""
Returns the number of possible single_linkage structures for `nlink` linkages and `nlink + 1` linkages, N(nlink + 1) / N(nlink)
```logratio_single_linkage(nlink, nrow, ncol)```
"""
function  logratio_single_linkage{G <: Integer}(nlink::G, nrow::G, ncol::G)
    return log(nrow - nlink) + log(ncol - nlink) - log(nlink + 1)
end

"""
logpenalty is subtracted so returned value is log(N(nlink + 1)) - log(N(nlink)) - penalty
"""
function  logratio_single_linkage{G <: Integer, T <: AbstractFloat}(nlink::G, nrow::G, ncol::G, logpenalty::T)
    return log(nrow - nlink) + log(ncol - nlink) - log(nlink + 1) - logpenalty
end


"""
Returns the number of possible single_linkage structures for `nlink1` linkages and `nlink2` linkages, N(nlink2) / N(nlink1)
```logratio_single_linkage(nlink1, nlink2, nrow, ncol)```
"""
function logratio_single_linkage{G <: Integer}(nlink1::G, nlink2::G, nrow::G, ncol::G)
    if nlink1 == nlink2
        return 0.0
    elseif nlink1 < nlink2
        out = 0.0
        nlink = nlink1
        while nlink < nlink2
            out += logratio_single_linkage(nlink, nrow, ncol)
            nlink += 1
        end
        return out
    else #nlink1 > nlink2
        out = 0.0
        nlink = nlink2
        while nlink < nlink1
            out += logratio_single_linkage(nlink, nrow, ncol)
            nlink += 1
        end
        return out
    end
end

function logratio_single_linkage{G <: Integer, T <: AbstractFloat}(nlink1::G, nlink2::G, nrow::G, ncol::G, logpenalty::T)
    if nlink1 == nlink2
        return 0.0
    elseif nlink1 < nlink2
        out = 0.0
        nlink = nlink1
        while nlink < nlink2
            out += logratio_single_linkage(nlink, nrow, ncol, logpenalty)
            nlink += 1
        end
        return out
    else #nlink1 > nlink2
        out = 0.0
        nlink = nlink2
        while nlink < nlink1
            out += logratio_single_linkage(nlink, nrow, ncol, logpenalty)
            nlink += 1
        end
        return out
    end
end

"""
Find mode of single linkage structure, number of linkages that leads to the most possible linkages, incases where the mode is tied as with 15 rows and 20 columns the smaller is reported.
"""
function mode_single_linkage{G <: Integer}(nrow::G, ncol::G)
    out = 0
    while logratio_single_linkage(out, nrow, ncol) > 0.0
        out += 1
    end
    return out
end

function mode_single_linkage{G <: Integer, T <: AbstractFloat}(nrow::G, ncol::G, logpenalty::T)
    out = 0
    while logratio_single_linkage(out, nrow, ncol, logpenalty) > 0.0
        out += 1
    end
    return out
end

"""
Sample a single_linkage matrix with `nlinks` linkages given `nrow` rows and `ncol` columns
`sampler_linkage_structure(nlinks, nrow, ncol)`
"""
function sampler_linkage_structure{G <: Integer}(nlinks::G, nrow::G, ncol::G)
    rows = StatsBase.sample(1:nrow, nlinks, replace = false)
    cols = StatsBase.sample(1:ncol, nlinks, replace = false)
    return rows, cols
end

"""
Count the number of linkages in on-diagonal blocks
"""
function count_diagonal_linkage{G <: Integer}(rows::Array{G, 1}, cols::Array{G, 1}, cumrows::Array{G, 1}, cumcols::Array{G, 1})
    dlinkages = zero(G)
    for (rr, cc) in zip(rows, cols)
        if searchsortedfirst(cumrows, rr) == searchsortedfirst(cumcols, cc)
            dlinkages += one(G)
        end
    end
    return dlinkages
end

"""
Count the number of linkages in off-diagonal blocks
"""
function count_offdiagonal_linkage{G <: Integer}(rows::Array{G, 1}, cols::Array{G, 1}, cumrows::Array{G, 1}, cumcols::Array{G, 1})
    offdlinkages = zero(G)
    for (rr, cc) in zip(rows, cols)
        if searchsortedfirst(cumrows, rr) != searchsortedfirst(cumcols, cc)
            offdlinkages += one(G)
        end
    end
    return offdlinkages
end

#logprobability_single_linkage{G <: Integer, T <: AbstractFloat}(nrow::G, ncol::G, logpenalty::T)
#
function rejection_sampler_single_linkage{T <: AbstractFloat}(loglinkpenalty::T, logProbs::Array{Float64, 1}, perm::Array{Int64, 1})
    while true
        nlinks = sample_logprobabilities(logProbs, perm) - 1 #check
        if log(rand()) < -nlinks * loglinkpenalty
            return nlinks
        end
    end
end

function rejection_sampler_single_linkage{G <: Integer, T <: AbstractFloat}(nrow::G, ncol::G, loglinkpenalty::T)
    logProbs = logprobability_single_linkage(nrow, ncol)
    perm = sortperm(logProbs, rev = true)
    return rejection_sampler_single_linkage(loglinkpenalty, logProbs, perm)
end

function rejection_sampler_single_linkage{G <: Integer, T <: AbstractFloat}(samplesize::G, nrow::G, ncol::G, loglinkpenalty::T)
    logProbs = logprobability_single_linkage(nrow, ncol)
    perm = sortperm(logProbs, rev = true)
    out = Array{G}(samplesize)
    for ii in 1:samplesize
        out[ii] =  rejection_sampler_single_linkage(loglinkpenalty, logProbs, perm)
    end
    return out
end

function rejection_sampler_single_linkage{G <: Integer, T <: AbstractFloat}(nrow::G, ncol::G, cumrows::Array{G, 1}, cumcols::Array{G, 1}, logProbs::Array{T, 1}, perm::Array{Int64, 1}, logoffdiagonalpenality::T)
        while true
        nlinks = sample_logprobabilities(logProbs, perm) - 1 #check
        rows, cols = sampler_linkage_structure(nlinks, nrow, ncol)
        offdlinks = count_offdiagonal_linkage(rows, cols, cumrows, cumcols)
        if log(rand()) < -offdlinks * logoffdiagonalpenality
            return nlinks, offdlinks
        end
    end
end


function rejection_sampler_single_linkage{G <: Integer, T <: AbstractFloat}(nrows::Array{G, 1}, ncols::Array{G, 1}, loglinkpenalty::T, logoffdiagonalpenality::T)
    cumrows = cumsum(nrows)
    cumcols = cumsum(ncols)
    nrow = cumrows[end]
    ncol = cumcols[end]
    logProbs = logprobability_single_linkage(nrow, ncol, loglinkpenalty)
    perm = sortperm(logProbs, rev = true)
    return rejection_sampler_single_linkage(nrow, ncol, cumrows, cumcols, logProbs, perm, logoffdiagonalpenality)
end

function rejection_sampler_single_linkage{G <: Integer, T <: AbstractFloat}(samplesize::Int64, nrows::Array{G, 1}, ncols::Array{G, 1}, loglinkpenalty::T, logoffdiagonalpenality::T)
    cumrows = cumsum(nrows)
    cumcols = cumsum(ncols)
    nrow = cumrows[end]
    ncol = cumcols[end]
    logProbs = logprobability_single_linkage(nrow, ncol, loglinkpenalty)
    perm = sortperm(logProbs, rev = true)
    
    outLinks = Array{G}(samplesize)
    outOffDLinks = Array{G}(samplesize)
    
    for ii in 1:samplesize
        outLinks[ii], outOffDLinks[ii] =  rejection_sampler_single_linkage(nrow, ncol, cumrows, cumcols, logProbs, perm, logoffdiagonalpenality)
    end
    return outLinks, outOffDLinks
end

function permutation_diagonalindex{G <: Integer}(k::G)
    idx = Array{G}(k)
    idx[1] = one(G)
    for ii in 2:k
        idx[ii] = idx[ii - 1] + k + one(G)
    end
    return StatsBase.sample(idx, k, replace = false)
end

function permutation_offdiagonalindex{G <: Integer}(k::G)
    idx = Array{G}(k * k - k)
    kk = one(G)
    for jj in 1:k
        for ii in 1:k
            if ii != jj                
                idx[kk] = sub2ind((k,k), ii, jj)
                kk += one(G)
            end
        end
    end
    return StatsBase.sample(idx, k * k - k, replace = false)
end

function permutation_blockindex{G <: Integer}(k::G)
    return StatsBase.sample(1:(k^2), k^2, replace = false)
end

function sampler_nlinks_single_linkage{G <: Integer}(nrow::G, ncol::G)
    if nrow == zero(G) || ncol == zero(G)
        return zero(Int64)
    else
        logP = logprobability_single_linkage(nrow, ncol)
        return sample_logprobabilities(logP) - 1
    end
end

function sampler_nlinks_single_linkage{G <: Integer, T <: AbstractFloat}(nrow::G, ncol::G, logpenalty::T)
    if nrow == zero(G) || ncol == zero(G)
        return zero(Int64)
    else
        logP = logprobability_single_linkage(nrow, ncol, logpenalty)
        return sample_logprobabilities(logP) - 1
    end
end

function sampler_blocklinks_single_linkage{G <: Integer}(nrows::Array{G,1}, ncols::Array{G,1})
    k = length(nrows)
    if k != length(ncols)
        error("number of rows blocks and column blocks must match")
    end

    #Comptue quantities
    openrows = copy(nrows)
    opencols = copy(ncols)
    blocklinks = zeros(Int64, k, k)

    #Sample on-diagonal links
    for idx in permutation_blockindex(k)
        brow, bcol = ind2sub((k,k), idx)
        nlinks = sampler_nlinks_single_linkage(openrows[brow], opencols[bcol])
        openrows[brow] -= nlinks
        opencols[bcol] -= nlinks
        blocklinks[brow, bcol] = nlinks
    end

    return blocklinks
end

function sampler_blocklinks_single_linkage{G <: Integer, T <: AbstractFloat}(nrows::Array{G,1}, ncols::Array{G,1}, logpenalty::T)
    k = length(nrows)
    if k != length(ncols)
        error("number of rows blocks and column blocks must match")
    end

    #Comptue quantities
    openrows = copy(nrows)
    opencols = copy(ncols)
    blocklinks = zeros(Int64, k, k)

    #Sample on-diagonal links
    for idx in permutation_blockindex(k)
        brow, bcol = ind2sub((k,k), idx)
        nlinks = sampler_nlinks_single_linkage(openrows[brow], opencols[bcol], logpenalty)
        openrows[brow] -= nlinks
        opencols[bcol] -= nlinks
        blocklinks[brow, bcol] = nlinks
    end

    return blocklinks
end


function sampler_stepblocklinks_single_linkage{G <: Integer}(nrows::Array{G,1}, ncols::Array{G,1})
    k = length(nrows)
    if k != length(ncols)
        error("number of rows blocks and column blocks must match")
    end

    #Comptue quantities
    openrows = copy(nrows)
    opencols = copy(ncols)
    onIdx = permutation_diagonalindex(k)
    offIdx = permutation_offdiagonalindex(k)
    blocklinks = zeros(Int64, k, k)

    #Sample on-diagonal links
    for idx in onIdx
        brow, bcol = ind2sub((k,k), idx)
        nlinks = sampler_nlinks_single_linkage(openrows[brow], opencols[bcol])
        openrows[brow] -= nlinks
        opencols[bcol] -= nlinks
        blocklinks[brow, bcol] = nlinks
    end

    #Sample off-diagonal links
    for idx in offIdx
        brow, bcol = ind2sub((k,k), idx)
        nlinks = sampler_nlinks_single_linkage(openrows[brow], opencols[bcol])
        openrows[brow] -= nlinks
        opencols[bcol] -= nlinks
        blocklinks[brow, bcol] = nlinks
    end
    return blocklinks
end

function sampler_stepblocklinks_single_linkage{G <: Integer, T <: AbstractFloat}(nrows::Array{G,1}, ncols::Array{G,1}, logpenalty::T)
    k = length(nrows)
    if k != length(ncols)
        error("number of rows blocks and column blocks must match")
    end

    #Comptue quantities
    openrows = copy(nrows)
    opencols = copy(ncols)
    onIdx = permutation_diagonalindex(k)
    offIdx = permutation_offdiagonalindex(k)
    blocklinks = zeros(Int64, k, k)

    #Sample on-diagonal links
    for idx in onIdx
        brow, bcol = ind2sub((k,k), idx)
        nlinks = sampler_nlinks_single_linkage(openrows[brow], opencols[bcol], logpenalty)
        openrows[brow] -= nlinks
        opencols[bcol] -= nlinks
        blocklinks[brow, bcol] = nlinks
    end

    #Sample off-diagonal links
    for idx in offIdx
        brow, bcol = ind2sub((k,k), idx)
        nlinks = sampler_nlinks_single_linkage(openrows[brow], opencols[bcol], logpenalty)
        openrows[brow] -= nlinks
        opencols[bcol] -= nlinks
        blocklinks[brow, bcol] = nlinks
    end
    return blocklinks
end


function sampler_stepblocklinks_single_linkage{G <: Integer, T <: AbstractFloat}(nrows::Array{G,1}, ncols::Array{G,1}, loglinkpenalty::T, logoffdiagonalpenality::T)
    k = length(nrows)
    if k != length(ncols)
        error("number of rows blocks and column blocks must match")
    end

    #Comptue quantities
    openrows = copy(nrows)
    opencols = copy(ncols)
    onIdx = permutation_diagonalindex(k)
    offIdx = permutation_offdiagonalindex(k)
    blocklinks = zeros(Int64, k, k)

    #Sample on-diagonal links
    for idx in onIdx
        brow, bcol = ind2sub((k,k), idx)
        nlinks = sampler_nlinks_single_linkage(openrows[brow], opencols[bcol], loglinkpenalty)
        openrows[brow] -= nlinks
        opencols[bcol] -= nlinks
        blocklinks[brow, bcol] = nlinks
    end

    #Sample off-diagonal links
    for idx in offIdx
        brow, bcol = ind2sub((k,k), idx)
        nlinks = sampler_nlinks_single_linkage(openrows[brow], opencols[bcol], loglinkpenalty + logoffdiagonalpenality)
        openrows[brow] -= nlinks
        opencols[bcol] -= nlinks
        blocklinks[brow, bcol] = nlinks
    end
    return blocklinks
end

function sampler_single_linkage{G <: Integer, V <: Integer}(blocklinks::Array{V, 2}, cumrows::Array{G, 1}, cumcols::Array{G, 1}; totallinks::V = sum(blocklinks))
    rowopen = trues(cumrows[end])
    colopen = trues(cumcols[end])
    rows = zeros(G, totallinks)
    cols = zeros(G, totallinks)
    addedlinks = 0
    for (jj, maxcols) in enumerate(cumcols)
        mincols = get(cumcols, jj - 1, 0) + 1
        for (ii, maxrows) in enumerate(cumrows)
            minrows = get(cumrows, ii - 1, 0) + 1

            #choose rows and cols to link
            addrows = StatsBase.sample((minrows:maxrows)[rowopen[minrows:maxrows]], blocklinks[ii, jj], replace = false)
            addcols = StatsBase.sample((mincols:maxcols)[colopen[mincols:maxcols]], blocklinks[ii, jj], replace = false)

            #add to linked rows and columns
            rows[(addedlinks + 1):(addedlinks + blocklinks[ii, jj])] = addrows
            cols[(addedlinks + 1):(addedlinks + blocklinks[ii, jj])] = addcols

            #mark selected rows and cols as linked
            rowopen[addrows] = false
            colopen[addcols] = false

            #add to total links
            addedlinks += blocklinks[ii, jj]
        end
    end
    return rows, cols
end

function sampler_single_linkage{G <: Integer}(nrows::Array{G, 1}, ncols::Array{G, 1})
    blocklinks = sampler_blocklinks_single_linkage(nrows, ncols)
    return sampler_single_linkage(blocklinks, cumsum(nrows), cumsum(ncols))
end

function sampler_single_linkage{G <: Integer, T <: AbstractFloat}(nrows::Array{G, 1}, ncols::Array{G, 1}, logpenalty::T)
    blocklinks = sampler_blocklinks_single_linkage(nrows, ncols, logpenalty)
    return sampler_single_linkage(blocklinks, cumsum(nrows), cumsum(ncols))
end

function sampler_step_single_linkage{G <: Integer}(nrows::Array{G, 1}, ncols::Array{G, 1})
    blocklinks = sampler_stepblocklinks_single_linkage(nrows, ncols)
    return sampler_single_linkage(blocklinks, cumsum(nrows), cumsum(ncols))
end

function sampler_step_single_linkage{G <: Integer, T <: AbstractFloat}(nrows::Array{G, 1}, ncols::Array{G, 1}, logpenalty::T)
    blocklinks = sampler_stepblocklinks_single_linkage(nrows, ncols, logpenalty)
    return sampler_single_linkage(blocklinks, cumsum(nrows), cumsum(ncols))
end

function sampler_step_single_linkage{G <: Integer, T <: AbstractFloat}(nrows::Array{G, 1}, ncols::Array{G, 1}, loglinkpenalty::T, logoffdiagonalpenality::T)
    blocklinks = sampler_stepblocklinks_single_linkage(nrows, ncols, loglinkpenalty, logoffdiagonalpenality)
    return sampler_single_linkage(blocklinks, cumsum(nrows), cumsum(ncols))
end
#=
"""
Recursively determine the log of the number of single linkages across non-overlapping blocks holding total number of linkages constant
"""
@memoize function logcount_block(totallink, nrows, ncols, upperbounds)
    if length(nrows) == 1
        tot = logfactorial(nrows[1]) + logfactorial(ncols[1])
        tot -= logfactorial(totallink)
        tot -= logfactorial(nrows[1] - totallink)
        tot -= logfactorial(ncols[1] - totallink) 
        return tot
    else
        minlinks = sum(upperbounds[2:end])
        maxlinks = min(totallink, nrows[1], ncols[1])
        logtotals = Array{Float64}(maxlinks - minlinks + 1)
        for (ii, nlink) in enumerate(minlinks:maxlinks)
            logtotals[ii] = logcount_block(nlink, nrows[1], ncols[1], upperbounds[1])
            + logcount_block(totallink - nlink, nrows[2:end], ncols[2:end], upperbounds[2:end])
        end
        return logsum(logtotals)
    end
end
=#
