#Example

import StatsBase
import Distributions

include("../src/matching_matrix.jl")
include("../src/logitnormal.jl")
include("../src/uniformsinglelinkage.jl")
include("../src/truncatedpoisson.jl")
include("../src/simrecords.jl")
include("../src/mcmc.jl")

pM = [0.8, 0.9, 0.75]
pU = [0.08, 0.02, .05]
srand(68259)
C = rand(UniformSingleLinkage(10, 10, 7))
data = simulate_singlelinkage_binary(C, pM, pU)

nones = countones(data)
Base.maximum(nones)
sum(nones .== 3)
sum(nones .== 2)
sum(nones .== 1)
sum(nones .== 0)

datatotable(data, C0)
#Initial Values
C0::MatchMatrix
M0::Array{AbstractFloat, 1}
U0::Array{AbstractFloat, 1}

#Evaluate densities
#logpdfC::Function
function lpC(C::MatchMatrix)
    lp1 = Distributions.logpdf(TrucatedPoisson(λ, 10), length(C.rows))
    lp2 = Distributions.logpdf(UniformSinglLinkage(10, 10, t), C)
    return lp1 + lp2
end

#logpdfM::Function #Distributions.logpdf(d::LogisticNormal, x::Real)
function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = x
    Distributions.logpdf.(d, γM)
end
#logpdfU::Function #Distributions.logpdf(d::LogisticNormal, x::Real)
function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = x
    Distributions.logpdf.(d, γU)
end

#loglikelihood::Function

#Transition Functions
#transitionC::Function #move_matchmatrix(M::MatchMatrix, p::AbstractFloat)
function transC(C::MatchMatrix)
    return move_matchmatrix(M, 0.5)
end
#transitionMU::Function #Distributions.rand(d::LogisticNormal)
function transMU{T <: AbstractFloat}(probs::Array{T, 1})
    dArray = LogisticNormal.(probs, 3.0)
    return rand.(dArray)
end

#Transition ratios
#transitionC_ratio::Function #ratio_pmove(M1::MatchMatrix, M2::MatchMatrix, p::AbstractFloat)
function transC_ratio(M1::MatchMatrix, M2::MatchMatrix)
    return ratio_pmove(M1, M2, 0.5)
end

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
