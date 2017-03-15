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
Returns the number of possible single_linkage structures for `nlink` linkages and `nlink + 1` linkages, N(nlink + 1) / N(nlink)
```logratio_single_linkage(nlink, nrow, ncol)```
"""
function  logratio_single_linkage{G <: Integer}(nlink::G, nrow::G, ncol::G)
    return log(nrow - nlink) + log(ncol - nlink) - log(nlink + 1)
end

"""
Returns the number of possible single_linkage structures for `nlink1` linkages and `nlink2` linkages, N(nlink2) / N(nlink1)
```logratio_single_linkage(nlink1, nlink2, nrow, ncol)```
"""
function logratio_single_linkage{G <: Integer}(nlink1::G, nlink2::G, nrow::G, ncol::G)
    if nlink1 == nlink2
        return 0
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
