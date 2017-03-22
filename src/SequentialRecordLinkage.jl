module SequentialRecordLinkage

import StatsBase
import Distributions
import Base: convert, copy, in, ==

export LogisticNormal,
    logit,
    logistic

export TruncatedPoisson
export VolumePoisson
export UniformSingleLinkage
export MatchMatrix,
    findindex,
    add_match!,
    remove_match!,
    empty_rows,
    empty_cols,
    ratio_pmove,
    match_pairs,
    move_matchmatrix,
    totalmatches

export BlockMatchMatrix,
    getblockrow,
    getblockcol,
    getrows,
    getcols,
    getindicies,
    getmatches,
    move_blockmatchmatrix,
    move_blockmatchmatrix_exclude,
    add_match!,
    add_match
    
export simulate_singlelinkage_binary, single_linkage_levels, gridtoarray

export data2table, loglikelihood_datatable, countones

export UnitKernelDensity, unitkde_slow, unitkde_tilted

export UnitKDEMixture, beta_mode

#MCMC Functions
export metropolis_hastings, metropolis_hastings_mixing
export metropolis_hastings_sample
export metropolis_hastings_twostep, metropolis_hastings_conditional_sample
export metropolis_hastings_ptwostep
export write_matchmatrix, write_probs, read_matchmatrix

#Functions for determining the prior
export logcount_single_linkage,
    logproportion_single_linkage,
    logprobability_single_linkage,
    logratio_single_linkage,
    mode_single_linkage,
    sampler_linkage_structure,
    count_diagonal_linkage,
    count_offdiagonal_linkage,
    rejection_sampler_single_linkage
export logsum, logfactorial
export sample_logprobabilities, rejection_sampler_logproportions

include("logisticnormal.jl")
include("truncatedpoisson.jl")
include("volumepoisson.jl")
include("matchmatrix.jl")
include("blockmatchmatrix.jl")
include("uniformsinglelinkage.jl")

include("simrecords.jl")
include("datatable.jl")
include("unitkerneldensity.jl")
include("unitkdemixture.jl")

include("mcmc.jl")
include("blockmcmc.jl")
include("stepmcmc.jl")
include("pstepmcmc.jl")
include("post_processing.jl")
include("single_linkage.jl")
include("utils.jl")
include("priorsamplers.jl")
end
