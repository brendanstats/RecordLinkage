#=
Rejection Samplers for Sampling from Various Priors
=#

"""
Sample from a set of probabilities, assume they sum to one, that have been logged.  Comparisons and addition done one a log scale to increase precision in the case of extrememly small probabilities.
"""
function sample_logprobabilities{G <: AbstractFloat}(logProbs::Array{G, 1})
    logCDF = log(rand(G))
    for (ii, logP) in enumerate(logProbs)
        if logP > logCDF
            return ii
        else
            logCDF = logP + log(exp(logCDF - logP) - one(logP))
        end
    end
end

function sample_logprobabilities{G <: AbstractFloat}(logProbs::Array{G, 1}, perm::Array{Int64, 1})
    logCDF = log(rand(G))
    for ii in perm
        logP = logProbs[ii]
        if logP > logCDF
            return ii
        else
            logCDF = logP + log(exp(logCDF - logP) - one(logP))
        end
    end
end

function sample_logprobabilities{G <: AbstractFloat, T <: Integer}(samplesize::T, logProbs::Array{G, 1}; sortprobs::Bool = false)
    out = Array{Int64}(samplesize)
    if sortprobs
        perm = sortperm(logProbs, rev = true)
        for ii in 1:samplesize
            out[ii] = sample_logprobabilities(logProbs, perm)
        end
    else
        for ii in 1:samplesize
            out[ii] = sample_logprobabilities(logProbs)
        end
    end
    return out
end

function rejection_sampler_logproportions{G <: AbstractFloat, T <: Integer}(logProportions::Array{G, 1};
                                                                            maxLogProportion::G = maximum(logProportions),
                                                                            n::T = length(logProportions),
                                                                            logM::G = 0.01)
    while true
        idx = StatsBase.sample(1:n)
        if log(rand()) < logProportions[idx] - maxLogProportion - logM
            return idx
        end
    end
end

function rejection_sampler_logproportions{G <: AbstractFloat, T <: Integer}(samplesize::T, logProportions::Array{G, 1};
                                                                            maxLogProportion::G = maximum(logProportions),
                                                                            n::T = length(logProportions),
                                                                            logM::G = 0.01)
    out = Array{T}(samplesize)
    for ii in 1:samplesize
        out[ii] = rejection_sampler_logproportions(logProportions, maxLogProportion = maxLogProportion, n = n, logM = logM)
    end
    return out
end



function draw_nlinks_uniform{G <: Integer, T <: AbstractFloat}(nrow::G, ncol::G, log::Array{T, 1})
    
end

function draw_nlinks_uniform{G <: Integer}(nrow::G, ncol::G)
    
end

function draw_nlinks_uniform{G <: Integer}(n::G, nrow::G, ncol::G)
    
end

function draw_uniform_single_linkage{G <: Integer}(nrow::G, ncol::G)
end
