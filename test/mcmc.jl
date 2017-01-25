#Example

import StatsBase
import Distributions

include("../src/matching_matrix.jl")
include("../src/logisticnormal.jl")
include("../src/uniformsinglelinkage.jl")
include("../src/truncatedpoisson.jl")
include("../src/volumepoisson.jl")
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

d = Distributions.Beta(5, 2)
Distributions.logpdf(d, pM)

#logpdfM::Function #Distributions.logpdf(d::LogisticNormal, x::Real)
function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = Distributions.Beta(5, 2)
    return sum(Distributions.logpdf(d, γM))
end

lpM(M0)

#logpdfU::Function #Distributions.logpdf(d::LogisticNormal, x::Real)
function lpU{T <: AbstractFloat}(γU::Array{T, 1})
    d = Distributions.Beta(2, 20)
    return sum(Distributions.logpdf(d, γU))
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
    d1 = LogisticNormal.(p1, 2.5)
    d2 = LogisticNormal.(p2, 2.5)
    logp = sum(Distributions.logpdf.(d2, p1)) - sum(Distributions.logpdf.(d1, p2))
    return exp(logp)
end

niter = 100000

srand(48397)
CArray, MArray, UArray = metropolis_hastings(niter, data, C0, M0, U0, lpC, lpM, lpU, loglikelihood_datatable, transC, transMU, transC_ratio, transMU_ratio)

nmatches, matchdist = totalmatches(CArray)

using RCall

R"library(ggplot2)"
R"library(tidyr)"
#R"library(gridExtra)"

@rput MArray UArray
@rput pM pU
R"ls()"
R"probs.df <- as.data.frame(cbind(MArray, UArray))";
R"names(probs.df) <- c('M1', 'M2', 'M3', 'U1', 'U2', 'U3')";
R"probs.df <- gather(probs.df, Comparison, P, M1:U3)";
R"genprob.df <- data.frame(Comparison = c('M1', 'M2', 'M3', 'U1', 'U2', 'U3'), genp = c(pM, pU))";

R"getwd()"

R"pdf('matching_posterior.pdf')"
R"ggplot(probs.df, aes(x = P)) + geom_density() + facet_wrap(~Comparison) +
 geom_vline(aes(xintercept = genp), genprob.df, color = 'red') +
 ggtitle('Posterior Matching Probabilities')"
R"dev.off()"


#R"par(mfrow = c(2, 3))"
#R"plot(m1, ylim = c(0,1))"
#R"plot(m2, ylim = c(0,1))"
#R"plot(m3, ylim = c(0,1))"
#R"plot(u1, ylim = c(0,1))"
#R"plot(u2, ylim = c(0,1))"
#R"plot(u3, ylim = c(0,1))"
#R"par(mfrow = c(1, 1))"

data1 = gridtoarray(data[1, :, :])
data2 = gridtoarray(data[2, :, :])
data3 = gridtoarray(data[3, :, :])
postdens = gridtoarray(matchdist)

@rput nmatches data1 data2 data3 postdens

#data processing
R"matches.df <- data.frame(matches = nmatches, timestep = 1:length(nmatches))";

R"compvec <- unlist(lapply(1:dim(data1)[1], function(n) paste0('(', data1[n, 3], ', ', data2[n, 3], ', ', data3[n, 3], ')')))";

R"postdens.df <- as.data.frame(postdens)";
R"names(postdens.df) <- c('row', 'col', 'postmean')";
R"postdens.df$comparison <- compvec"


#make plots
R"ggplot(postdens.df) + geom_raster(aes(x = col, y = row, fill = postmean)) +
geom_text(aes(x = col, y = row, label = comparison), size = 3)"

R"hist(nmatches)"
R"plot(nmatches)"

#transitionMU::Function #Distributions.rand(d::LogisticNormal)
function transM{T <: AbstractFloat}(probs::Array{T, 1})
    dArray = LogisticNormal.(probs, 1.1)
    return rand.(dArray)
end

function transM_ratio{T <: AbstractFloat}(p1::Array{T, 1}, p2::Array{T, 1})
    if length(p1) != length(p2)
        error("probability vector lengths must match")
    end
    d1 = LogisticNormal.(p1, 1.1)
    d2 = LogisticNormal.(p2, 1.1)
    logp = sum(Distributions.logpdf.(d2, p1)) - sum(Distributions.logpdf.(d1, p2))
    return exp(logp)
end

#transitionMU::Function #Distributions.rand(d::LogisticNormal)
function transU{T <: AbstractFloat}(probs::Array{T, 1})
    dArray = LogisticNormal.(probs, 0.8)
    return rand.(dArray)
end

function transU_ratio{T <: AbstractFloat}(p1::Array{T, 1}, p2::Array{T, 1})
    if length(p1) != length(p2)
        error("probability vector lengths must match")
    end
    d1 = LogisticNormal.(p1, 0.8)
    d2 = LogisticNormal.(p2, 0.8)
    logp = sum(Distributions.logpdf.(d2, p1)) - sum(Distributions.logpdf.(d1, p2))
    return exp(logp)
end

#Effect of prior on C...
function lpC(C::MatchMatrix)
    return -length(C.rows) * 0.6
end
CArray2, MArray2, UArray2, tC, tM, tU = metropolis_hastings_mixing(100000, data, C0, M0, U0, lpC, lpM, lpU, loglikelihood_datatable, transC, transM, transU, transC_ratio, transM_ratio, transU_ratio)

function lpC(C::MatchMatrix)
    return -length(C.rows) * 1.8
end
CArray3, MArray3, UArray3, tC, tM, tU = metropolis_hastings_mixing(100000, data, C0, M0, U0, lpC, lpM, lpU, loglikelihood_datatable, transC, transM, transU, transC_ratio, transM_ratio, transU_ratio)

function lpC(C::MatchMatrix)
    return -length(C.rows) * 3.0
end
CArray4, MArray4, UArray4, tC, tM, tU = metropolis_hastings_mixing(100000, data, C0, M0, U0, lpC, lpM, lpU, loglikelihood_datatable, transC, transM, transU, transC_ratio, transM_ratio, transU_ratio)


#using RCall
#R"library(ggplot2)"
#R"library(tidyr)"
#R"library(latex2exp)"

@rput MArray2 UArray2 MArray3 UArray3 MArray4 UArray4
@rput pM pU

R"probs2.df <- as.data.frame(cbind(1:nrow(MArray2), MArray2, UArray2))";
R"probs3.df <- as.data.frame(cbind(1:nrow(MArray3), MArray3, UArray3))";
R"probs4.df <- as.data.frame(cbind(1:nrow(MArray4), MArray4, UArray4))";

R"names(probs2.df) <- c('Iter', 'M1', 'M2', 'M3', 'U1', 'U2', 'U3')";
R"names(probs3.df) <- c('Iter', 'M1', 'M2', 'M3', 'U1', 'U2', 'U3')";
R"names(probs4.df) <- c('Iter', 'M1', 'M2', 'M3', 'U1', 'U2', 'U3')";

R"probs2.df <- gather(probs2.df, Comparison, P, M1:U3)";
R"probs3.df <- gather(probs3.df, Comparison, P, M1:U3)";
R"probs4.df <- gather(probs4.df, Comparison, P, M1:U3)";

R"genprob.df <- data.frame(Comparison = c('M1', 'M2', 'M3', 'U1', 'U2', 'U3'), genp = c(pM, pU))";

R"pdf('matching_posterior.pdf')"
R"ggplot(probs2.df, aes(x = P)) + geom_density() + facet_wrap(~Comparison) +
 geom_vline(aes(xintercept = genp), genprob.df, color = 'red') +
 ggtitle(TeX('Posterior Matching Probabilities $\\theta = 0.6$'))"

R"ggplot(probs3.df, aes(x = P)) + geom_density() + facet_wrap(~Comparison) +
 geom_vline(aes(xintercept = genp), genprob.df, color = 'red') +
 ggtitle(TeX('Posterior Matching Probabilities $\\theta = 1.8$'))"

R"ggplot(probs4.df, aes(x = P)) + geom_density() + facet_wrap(~Comparison) +
 geom_vline(aes(xintercept = genp), genprob.df, color = 'red') +
 ggtitle(TeX('Posterior Matching Probabilities $\\theta = 3.0$'))"

R"ggplot(probs2.df, aes(x = Iter, y = P)) + geom_line() + facet_wrap(~Comparison) +
 geom_hline(aes(yintercept = genp), genprob.df, color = 'red') +
 ggtitle(TeX('Posterior Matching Probabilities $\\theta = 0.6$'))"
R"dev.off()"

data1 = gridtoarray(data[1, :, :])
data2 = gridtoarray(data[2, :, :])
data3 = gridtoarray(data[3, :, :])
nmatches2, matchdist2 = totalmatches(CArray2)
postdens2 = gridtoarray(matchdist2)

nmatches3, matchdist3 = totalmatches(CArray3)
postdens3 = gridtoarray(matchdist3)

nmatches4, matchdist4 = totalmatches(CArray4)
postdens4 = gridtoarray(matchdist4)

@rput data1 data2 data3 nmatches2 nmatches3 nmatches4 postdens2 postdens3 postdens4

#data processing
R"matches2.df <- data.frame(matches = nmatches2, timestep = 1:length(nmatches2))";
R"matches3.df <- data.frame(matches = nmatches3, timestep = 1:length(nmatches3))";
R"matches4.df <- data.frame(matches = nmatches4, timestep = 1:length(nmatches4))";

R"compvec <- unlist(lapply(1:dim(data1)[1], function(n) paste0('(', data1[n, 3], ', ', data2[n, 3], ', ', data3[n, 3], ')')))";

R"postdens2.df <- as.data.frame(postdens2)";
R"postdens3.df <- as.data.frame(postdens3)";
R"postdens4.df <- as.data.frame(postdens4)";

R"names(postdens2.df) <- c('row', 'col', 'postmean')";
R"names(postdens3.df) <- c('row', 'col', 'postmean')";
R"names(postdens4.df) <- c('row', 'col', 'postmean')";

R"postdens2.df$comparison <- compvec"
R"postdens3.df$comparison <- compvec"
R"postdens4.df$comparison <- compvec"

#make plots
R"pdf('grid_posterior.pdf')"
R"ggplot(postdens2.df) + geom_raster(aes(x = col, y = row, fill = postmean)) +
geom_text(aes(x = col, y = row, label = comparison), size = 3) +
 ggtitle(TeX('Posterior Matching Probabilities $\\theta = 0.6$'))"
R"ggplot(postdens3.df) + geom_raster(aes(x = col, y = row, fill = postmean)) +
geom_text(aes(x = col, y = row, label = comparison), size = 3) +
 ggtitle(TeX('Posterior Matching Probabilities $\\theta = 1.8$'))"
R"ggplot(postdens4.df) + geom_raster(aes(x = col, y = row, fill = postmean)) +
geom_text(aes(x = col, y = row, label = comparison), size = 3) +
 ggtitle(TeX('Posterior Matching Probabilities $\\theta = 3.0$'))"
R"hist(nmatches2, main = TeX('Posterior Matching Probabilities $\\theta = 0.6$'))"
R"hist(nmatches3, main = TeX('Posterior Matching Probabilities $\\theta = 1.8$'))"
R"hist(nmatches4, main = TeX('Posterior Matching Probabilities $\\theta = 3.0$'))"
R"dev.off()"
