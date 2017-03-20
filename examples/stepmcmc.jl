using SequentialRecordLinkage, RCall

#Define true values and generate data
srand(68259)

data, C, pM, pU =  single_linkage_levels(100, 40, 40, [2, 4, 4, 5], [0.2, 0.12, 0.2, 0.05], blocking = true)

nones = sum(data, 3)[:, :, 1]
gridarray = gridtoarray(nones)

#Compute reasonable intial values
keep = gridarray[:, 3] .== 4
rows = gridarray[keep, 1]
cols = gridarray[keep, 2]
remove = (7, 8, 9, 14, 15)
deleteat!(rows, remove)
deleteat!(cols, remove)

#Set Initial Values
C0 = MatchMatrix(rows, cols, 40, 40)
M0 = [0.95, 0.95, 0.95]
U0 = vec((sum(data[:, :, 2:4], 1:2) .- length(rows)) ./ (40 * 40 - length(rows)))

b1rows = sum(data[:, 1, 1])
b1cols = sum(data[1, :, 1])
GM0 = GridMatchMatrix([b1rows, 40 - b1rows], [b1cols, 40 - b1cols], C0)

#=
Standard algorithm
=#

function marginal_logcoef{G <: Integer, T <: AbstractFloat}(l::G, nrow::G, ncol::G, θ::T)
    if l == 0
        return -(sum(log(nrow)) + sum(log(ncol)))
    else
        return -θ + log(nrow - l + 1) + log(ncol - l + 1) -log(l)
    end
end


#Log prior density functions for probabilities
function lpC(C::MatchMatrix)
    θ = 0.6
    L11 = 0
    L12 = 0
    L21 = 0
    L22 = 0
    for (rr, cc) in zip(C.rows, C.cols)
        if rr <= b1rows
            if cc <= b1cols
                L11 += 1
            else
                L12 += 1
            end
        else
            if cc <= b1cols
                L21 += 1
            else
                L22 += 1
            end
        end
    end

    #Compute normalization constant for L12
    nrow12 = b1rows - L11
    ncol12 = C.ncol - b1cols - L22
    maxL12 = min(nrow12, ncol12)
    marginalLogCoef12 = Array{Float64}(maxL12 + 1)
    for ll in 0:maxL12
        marginalLogCoef12[ll + 1] = marginal_logcoef(ll, nrow12, ncol12, θ)
    end
    logCoef12 = cumsum(marginalLogCoef12)
    logNorm12 = logsum(logCoef12) + sum(log(1:nrow12)) + sum(log(1:ncol12))

    #Compute normalization constant for L21
    nrow21 = C.nrow - b1rows - L22
    ncol21 = b1cols - L11
    maxL21 = min(nrow21, ncol21)
    marginalLogCoef21 = Array{Float64}(maxL21 + 1)
    for ll in 0:maxL21
        marginalLogCoef21[ll + 1] = marginal_logcoef(ll, nrow21, ncol21, θ)
    end
    logCoef21 = cumsum(marginalLogCoef21)
    logNorm21 = logsum(logCoef21) + sum(log(1:nrow21)) + sum(log(1:ncol21))

    #return -length(C.rows) * 0.6
    return -length(C.rows) * θ - logNorm12 - logNorm21
end

function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = Distributions.Beta(5, 2)
    return sum(Distributions.logpdf(d, γM))
end

function lpU{T <: AbstractFloat}(γU::Array{T, 1})
    d = Distributions.Beta(2, 20)
    return sum(Distributions.logpdf(d, γU))
end

#Transition functions and probability ratios for standard algorithm
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

function transU{T <: AbstractFloat}(probs::Array{T, 1})
    dArray = LogisticNormal.(probs, 0.1)
    return rand.(dArray)
end

function transU_ratio{T <: AbstractFloat}(p1::Array{T, 1}, p2::Array{T, 1})
    if length(p1) != length(p2)
        error("probability vector lengths must match")
    end
    d1 = LogisticNormal.(p1, 0.1)
    d2 = LogisticNormal.(p2, 0.1)
    logp = sum(Distributions.logpdf.(d2, p1)) - sum(Distributions.logpdf.(d1, p2))
    return exp(logp)
end

function transC(C::MatchMatrix)
    return move_matchmatrix(C, 0.5)
end

function transC_ratio(M1::MatchMatrix, M2::MatchMatrix)
    return ratio_pmove(M1, M2, 0.5)
end

#Run standard algorithm
niter = 1000000
srand(48397)

@time CArray, MArray, UArray, chgC, chgM, chgU = metropolis_hastings_mixing(niter,
                                                                      data[:, :, 2:4],
                                                                      C0,
                                                                      M0,
                                                                      U0,
                                                                      lpC,
                                                                      lpM,
                                                                      lpU,
                                                                      loglikelihood_datatable,
                                                                      transC,
                                                                      transM,
                                                                      transU,
                                                                      transC_ratio,
                                                                      transM_ratio,
                                                                      transU_ratio)

write_matchmatrix("singlematchprior_results.txt", CArray)
write_probs("singleprobprior_results.txt", MArray, UArray)

#write_matchmatrix("singlematch_results.txt", CArray)
#write_probs("singleprob_results.txt", MArray, UArray)

srand(56969)

@time FCArray, FMArray, FUArray, chgC, chgM, chgU = metropolis_hastings_mixing(niter,
                                                                      data,
                                                                      C0,
                                                                      [.99; M0],
                                                                      [0.01; U0],
                                                                      lpC,
                                                                      lpM,
                                                                      lpU,
                                                                      loglikelihood_datatable,
                                                                      transC,
                                                                      transM,
                                                                      transU,
                                                                      transC_ratio,
                                                                      transM_ratio,
                                                                      transU_ratio)


write_matchmatrix("fullmatch_results.txt", FCArray)
write_probs("fullprob_results.txt", FMArray, FUArray)


#=
Step Algorithm
=#

#prior on the grid match matrix (just need something proportional)
function lpGM(grows::Array{Int64, 1}, gcols::Array{Int64, 1}, GM::GridMatchMatrix)
    θ = 0.6
    l = 0
    for (ii, jj) in zip(grows, gcols)
        l += length(GM.grid[ii, jj].rows)
    end
    return -θ * l
end

#transition functions with transition ratios
function transGM{G <: Integer}(grows::Array{G, 1}, gcols::Array{G, 1}, GM::GridMatchMatrix)
    return move_gridmatchmatrix(grows, gcols, GM, 0.5)
end

function transMGrid{T <: AbstractFloat}(probs::Array{T, 1})
    d1 = LogisticNormal.(probs, 1.05)
    probsNew = rand.(d1)
    d2 = LogisticNormal.(probsNew, 1.05)
    logp = sum(Distributions.logpdf.(d2, probs)) - sum(Distributions.logpdf.(d1, probsNew))
    return probsNew, exp(logp)
end

function transUGrid{T <: AbstractFloat}(probs::Array{T, 1})
    d1 = LogisticNormal.(probs, 0.1)
    probsNew = rand.(d1)
    d2 = LogisticNormal.(probsNew, 0.1)
    logp = sum(Distributions.logpdf.(d2, probs)) - sum(Distributions.logpdf.(d1, probsNew))
    return probsNew, exp(logp)
end


nsamples = 1000
niter1 = 1000000
niter2 = 10000

grows1 = [1, 2]
gcols1 = [1, 2]
grows2 = [1, 2]
gcols2 = [2, 1]
θ = 0.1
p = 0.1

srand(29171)

outC, outM, outU = metropolis_hastings_twostep(nsamples,
                                               niter1,
                                               niter2,
                                               data[:, :, 2:4],
                                               grows1,
                                               gcols1,
                                               grows2,
                                               gcols2,
                                               GM0,
                                               M0,
                                               U0,
                                               lpGM,
                                               lpM,
                                               lpU,
                                               loglikelihood_datatable,
                                               transGM,
                                               transMGrid,
                                               transUGrid,
                                               θ,
                                               p,
                                               nGM = 1,
                                               nM = 1,
                                               nU = 1)

write_matchmatrix("stepmatch_results.txt", outC)
write_probs("stepprob_results.txt", outM, outU)

#singleC, singleSamples = read_matchmatrix("singlematch_results.txt")
#fullC, fullSamples = read_matchmatrix("fullmatch_results.txt")
#stepC, stepSamples = read_matchmatrix("stepmatch_results.txt")

singleC, singleSamples = read_matchmatrix("singlematchprior_results.txt")
fullC, fullSamples = read_matchmatrix("fullmatch_results.txt")
stepC, stepSamples = read_matchmatrix("stepmatch_results.txt")

singleProp = singleC ./ singleSamples
fullProp = fullC ./ fullSamples
stepProp = stepC ./ stepSamples

singlePlot = gridtoarray(singleProp)
fullPlot = gridtoarray(fullProp)
stepPlot = gridtoarray(stepProp)

linkPlot = hcat(singlePlot, fullPlot[:, 3], stepPlot[:, 3])

R"library(ggplot2)"
R"library(tidyr)"
R"library(dplyr)"

@rput linkPlot

R"df <- as.data.frame(linkPlot)"
R"names(df) <- c('row', 'col', 'standard', 'full', 'step')"
R"fulldf <- mutate(df, 'standard - step' = standard - step, 'full - step' = full - step, 'standard - full' = standard - full)"

R"df1 <- fulldf %>% select(row, col, standard, full, step) %>% gather(method, proportion, -row, -col)"
R"df2 <- fulldf %>% select(row, col, contains('-')) %>% gather(method, proportion, -row, -col)"

R"pdf('linkage_comparison_stepmcmc.pdf')"
R"ggplot(df1, aes(x = col, y = row)) +
 geom_tile(aes(fill = proportion)) +
 geom_hline(yintercept = 21.5) +
 geom_vline(xintercept = 23.5) +
 scale_y_reverse() +
 facet_wrap(~method, ncol = 3) + 
 ggtitle('Posterior Linkage Proportions') +
 theme(plot.title = element_text(hjust = 0.5), legend.position = 'bottom')"

R"ggplot(df2, aes(x = col, y = row)) +
 geom_tile(aes(fill = proportion)) +
 geom_hline(yintercept = 21.5) +
 geom_vline(xintercept = 23.5) +
 scale_fill_gradient2() +
 scale_y_reverse() +
 facet_wrap(~method, ncol = 3) + 
 ggtitle('Difference in Posterior Linkage Proportions') +
 theme(plot.title = element_text(hjust = 0.5), legend.position = 'bottom')"
R"dev.off()"

singleProbs, spLabels = readdlm("singleprob_results.txt", '\t', header = true)
spLabels = vec(spLabels)
spLabels = String.(spLabels)

@rput spLabels
@rput singleProbs
R"sp.df <- as.data.frame(singleProbs)";
R"names(sp.df) <- spLabels"
R"sp.df <- gather(sp.df, probability, value, -index)";

R"ggplot(sp.df, aes(value)) +
 geom_density() +
 facet_wrap(~probability, ncol = 3, scale = 'free_y') +
 ggtitle('Posterior Distribution of Matching Probabilities\nJoint Matrix') +
 xlab('Matching Probability') + 
 theme(plot.title = element_text(hjust = 0.5))"


jointProbs, jointLabels = readdlm("stepprob_results.txt", '\t', header = true)
jointLabels = vec(jointLabels)
jointLabels = String.(jointLabels)

@rput jointLabels jointProbs
R"joint.df <- as.data.frame(jointProbs)";
R"names(joint.df) <- jointLabels"
R"joint.df <- gather(joint.df, probability, value, -index)";

R"ggplot(joint.df, aes(value)) +
 geom_density() +
 facet_wrap(~probability, ncol = 3, scale = 'free_y') +
 ggtitle('Posterior Distribution of Matching Probabilities\nJoint Matrix') +
 xlab('Matching Probability') + 
 theme(plot.title = element_text(hjust = 0.5))"
