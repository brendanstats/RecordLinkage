module SequentialRecordLinkage

import StatsBase
import Distributions
import HDF5
import Base.convert, Base.copy, Base.in, Base.==

#Computational Functions
export logsum, logfactorial, grid2array, switchidx, switchidx!

#Define new types
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
    getrowempty,
    getcolempty,
    getemptyrows,
    getemptycols,
    getblocknlinks,
    getnrowsremaining,
    getncolsremaining,
    add_match!,
    add_match,
    remove_match!,
    move_blockmatchmatrix,
    move_blockmatchmatrix_exclude

#Distrbution related functions
export LogisticNormal,
    logit,
    logistic
export TruncatedPoisson
export VolumePoisson
export UniformSingleLinkage
export sample_logprobabilities, rejection_sampler_logproportions
export logcount_single_linkage,
    logproportion_single_linkage,
    logprobability_single_linkage,
    logratio_single_linkage,
    mode_single_linkage,
    expected_nlinks_single_linkage,
    sampler_linkage_structure,
    matrix_logprobability_single_linkage,
    count_diagonal_linkage,
    count_offdiagonal_linkage,
    rejection_sampler_single_linkage,
    permutation_diagonalindex,
    permutation_offdiagonalindex,
    permutation_blockindex,
    sampler_nlinks_single_linkage,
    sampler_blocklinks_single_linkage,
    sampler_stepblocklinks_single_linkage,
    sampler_single_linkage,
    sampler_step_single_linkage,
    logprobability_blocknlinks_single_linkage

#RL Mechanics functions
export simulate_singlelinkage_binary, single_linkage_levels,
    estimate_C0, estimate_M0, estimate_U0
export data2table, loglikelihood_datatable, countones
export UnitKernelDensity, unitkde_slow, unitkde_tilted
export UnitKDEMixture, beta_mode

#MCMC Functions
export metropolis_hastings, metropolis_hastings_mixing
export metropolis_hastings_sample
export metropolis_hastings_permutation, metropolis_hastings_permutation_sample

export metropolis_hastings_twostep, metropolis_hastings_conditional_sample
export metropolis_hastings_ptwostep

export write_matchmatrix, write_probs, read_matchmatrix, writemhchains_h5

#Purely computational functions
include("utils.jl")

#New Types
include("matchmatrix.jl")
include("blockmatchmatrix.jl")

#Distributions related functions
include("logisticnormal.jl")
include("truncatedpoisson.jl")
include("volumepoisson.jl")
include("uniformsinglelinkage.jl")
include("priorsamplers.jl")
include("single_linkage.jl")

#RL Mechanics Functions
include("simrecords.jl")
include("datatable.jl")
include("unitkerneldensity.jl")
include("unitkdemixture.jl")

#MCMC Functions
include("mcmc.jl")
include("blockmcmc.jl")
include("blockpermmcmc.jl")

include("stepmcmc.jl")
include("pstepmcmc.jl")

include("post_processing.jl")
end
