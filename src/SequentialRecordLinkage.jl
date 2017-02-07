module SequentialRecordLinkage

import StatsBase
import Distributions
import Base: convert, copy, in

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

export GridMatchMatrix,
    getgridrow,
    getgridcol,
    getrows,
    getcols,
    getindicies,
    getmatches,
    move_gridmatchmatrix,
    move_gridmatchmatrix_exclude
    
export simulate_singlelinkage_binary,
    gridtoarray

export datatotable,
    loglikelihood_datatable,
    countones

export UnitKernelDensity,
    unitkde_slow,
    unitkde_interpolate,
    unitkde_tilted

export metropolis_hastings,
    metropolis_hastings_mixing

include("logisticnormal.jl")
include("truncatedpoisson.jl")
include("volumepoisson.jl")
include("matchmatrix.jl")
include("gridmatchmatrix.jl")
include("uniformsinglelinkage.jl")
include("simrecords.jl")
include("datatable.jl")
include("unitkerneldensity.jl")
include("mcmc.jl")
include("gridmcmc.jl")

end
