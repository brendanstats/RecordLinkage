#Example

import StatsBase
import Distributions

include("../src/matching_matrix.jl")
include("../src/logitnormal.jl")
include("../src/uniformsinglelinkage.jl")
include("../src/truncatedpoisson.jl")
include("../src/simrecords.jl")
include("../src/mcmc.jl")

pM = [0.8, 0.9, 0.7]
pU = [0.08, 0.02, .05]
srand(68259)
C = rand(UniformSingleLinkage(10, 10, 7))
data = simulate_singlelinkage_binary(C, pM, pU)

nones = countones(data)
C0 = MatchMatrix([2, 4, 5, 6, 8,10], [7, 6, 5, 1, 4, 9], 10, 10)

datatotable(data, C0)[1, :, :]
datatotable(data, C0)[2, :, :]
datatotable(data, C0)[3, :, :]

#Initial Values
M0 = [0.99, 0.99, 0.99]
U0 = [1. - 87. / 94., 1. - 92. / 94., 1. - 92. / 94.]

#Evaluate densities
#logpdfC::Function
function lpC(C::MatchMatrix)
    lp1 = Distributions.logpdf(TruncatedPoisson(6.0, 10), length(C.rows))
    lp2 = Distributions.logpdf(UniformSingleLinkage(10, 10, length(C.rows)), C)
    return lp1 + lp2
end

lpC(C0)

#logpdfM::Function #Distributions.logpdf(d::LogisticNormal, x::Real)
function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = x
    Distributions.logpdf.(d, γM)
end

lpM(M0)

#logpdfU::Function #Distributions.logpdf(d::LogisticNormal, x::Real)
function lpU{T <: AbstractFloat}(γU::Array{T, 1})
    d = x
    Distributions.logpdf.(d, γU)
end

lpU(U0)
#loglikelihood::Function

#Transition Functions
#transitionC::Function #move_matchmatrix(M::MatchMatrix, p::AbstractFloat)
function transC(C::MatchMatrix)
    return move_matchmatrix(C, 0.5)
end

transC(C0)

#transitionMU::Function #Distributions.rand(d::LogisticNormal)
function transMU{T <: AbstractFloat}(probs::Array{T, 1})
    dArray = LogisticNormal.(probs, 2.5)
    return rand.(dArray)
end

transMU(M0)
transMU(U0)

#Transition ratios
#transitionC_ratio::Function #ratio_pmove(M1::MatchMatrix, M2::MatchMatrix, p::AbstractFloat)
function transC_ratio(M1::MatchMatrix, M2::MatchMatrix)
    return ratio_pmove(M1, M2, 0.5)
end

transC_ratio(C0, transC(C0))

#transitionMU_ratio::Function #Distributions.logpdf(d::LogisticNormal, x::Real)
function transMU_ratio{T <: AbstractFloat}(p1::Array{T, 1}, p2::Array{T, 1})
    if length(p1) != length(p2)
        error("probability vector lengths must match")
    end
    d1 = LogisticNormal.(p1, 3.0)
    d2 = LogisticNormal.(p2, 3.0)
    logp = sum(Distributions.logpdf.(d2, p1)) - sum(Distributions.logpdf.(d1, p2))
    return exp(logp)
end

niter = 20

metropolis_hastings(niter, data, C0, M0, U0, lpC, lpM, lpU, loglikelihood_datatable, transC, transMU, transC_ratio, transMU_ratio)
