import StatsBase
import Distributions

include("../src/matching_matrix.jl")
include("../src/gridmatchingmatrix.jl")
include("../src/logisticnormal.jl")
include("../src/uniformsinglelinkage.jl")
include("../src/truncatedpoisson.jl")
include("../src/volumepoisson.jl")
include("../src/simrecords.jl")
include("../src/mcmc.jl")
include("../src/mcmc_grid.jl")

pM = [0.8, 0.9, 0.68]
pU = [0.15, 0.08, .45]
srand(68259)
C = rand(UniformSingleLinkage(40, 40, 27))
data = simulate_singlelinkage_binary(C, pM, pU)
database = [ones(Int8, 20, 20) zeros(Int8, 20, 20); zeros(Int8, 20, 20) ones(Int8, 20, 20)]
nones = countones(data)

#Log prior densities

#M probabilities
function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = Distributions.Beta(5, 2)
    return sum(Distributions.logpdf(d, γM))
end

#U probabilities
function lpU{T <: AbstractFloat}(γU::Array{T, 1})
    d = Distributions.Beta(2, 20)
    return sum(Distributions.logpdf(d, γU))
end

#Transition Functions
function transC(C::MatchMatrix)
    return move_matchmatrix(C, 0.5)
end

function transMU{T <: AbstractFloat}(probs::Array{T, 1})
    dArray = LogisticNormal.(probs, 2.5)
    return rand.(dArray)
end

#Transition Ratio
function transC_ratio(M1::MatchMatrix, M2::MatchMatrix)
    return ratio_pmove(M1, M2, 0.5)
end

function transMU_ratio{T <: AbstractFloat}(p1::Array{T, 1}, p2::Array{T, 1})
    if length(p1) != length(p2)
        error("probability vector lengths must match")
    end
    d1 = LogisticNormal.(p1, 2.5)
    d2 = LogisticNormal.(p2, 2.5)
    logp = sum(Distributions.logpdf.(d2, p1)) - sum(Distributions.logpdf.(d1, p2))
    return exp(logp)
end

niter = 100000

srand(48397)
CArray, MArray, UArray = metropolis_hastings(niter, data, C0, M0, U0, lpC, lpM, lpU, loglikelihood_datatable, transC, transMU, transC_ratio, transMU_ratio)


#Blocking Case
function transGM{G <: Integer}(grows::Array{G, 1}, gcols::Array{G, 1}, GM::GridMatchMatrix)
    return move_matchmatrix(C, 0.5)
end

function transMgrid{T <: AbstractFloat}(probs::Array{T, 1})
    d1 = LogisticNormal.(probs, 1.1)
    probsNew = rand.(d1)
    d2 = LogisticNormal.(probsNew, 1.1)
    logp = sum(Distributions.logpdf.(d2, probs)) - sum(Distributions.logpdf.(d1, probsNew))
    return probsNew, exp(logp)
end

function transUgrid{T <: AbstractFloat}(probs::Array{T, 1})
    d1 = LogisticNormal.(probs, 0.8)
    probsNew = rand.(d1)
    d2 = LogisticNormal.(probsNew, 0.8)
    logp = sum(Distributions.logpdf.(d2, probs)) - sum(Distributions.logpdf.(d1, probsNew))
    return probsNew, exp(logp)
end
