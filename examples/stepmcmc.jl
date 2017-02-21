using SequentialRecordLinkage, RCall

#Define true values and generate data
pM = [0.8, 0.9, 0.68]
pU = [0.15, 0.08, .45]
srand(68259)
C = rand(UniformSingleLinkage(40, 40, 27))
data = simulate_singlelinkage_binary(C, pM, pU)
database = [ones(Int8, 20, 20) zeros(Int8, 20, 20); zeros(Int8, 20, 20) ones(Int8, 20, 20)]
nones = countones(data)
gridarray = gridtoarray(nones)

#Compute reasonable intial values
rows = gridarray[gridarray[:, 3] .== 3, 1]
cols = gridarray[gridarray[:, 3] .== 3, 2]
deleteat!(rows, (2, 10, 11, 12, 14, 21, 25, 26))
deleteat!(cols, (2, 10, 11, 12, 14, 21, 25, 26))

#Set Initial Values
C0 = MatchMatrix(rows, cols, 40, 40)
M0 = [0.99, 0.99, 0.99]
U0 = vec((sum(data, 2:3) .- length(rows)) ./ (40 * 40 - length(rows)))
GM0 = GridMatchMatrix([20,20], [20,20], C0)

#=
Standard algorithm
=#

#Log prior density functions for probabilities
function lpC(C::MatchMatrix)
    return -length(C.rows) * 0.6
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
                                                                      data,
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

write_matchmatrix("singlematch_results.txt", CArray)
write_probs("singleprob_results.txt", MArray, UArray)

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
                                               data,
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

singleC, singleSamples = read_matchmatrix("singlematch_results.txt")
stepC, stepSamples = read_matchmatrix("stepmatch_results.txt")


singleProp = singleC ./ singleSamples
stepProp = stepC ./ stepSamples

singlePlot = gridtoarray(singleProp)
stepPlot = gridtoarray(stepProp)

R"library(ggplot2)"

@rput singlePlot stepPlot

R"df1 <- as.data.frame(singlePlot)"
R"names(df1) <- c('row', 'col', 'proportion')"
R"df1$type <- 'standard'"

R"df2 <- as.data.frame(stepPlot)"
R"names(df2) <- c('row', 'col', 'proportion')"
R"df2$type <- 'step'"

R"df3 <- df2"
R"df3$proportion <- df1$proportion - df2$proportion"
R"df3$type <- 'difference'"

R"plotdf <- rbind(df1, df2)"

R"pdf('comparison_stepmcmc.pdf')"
R"ggplot(plotdf, aes(x = col, y = row)) + geom_tile(aes(fill = proportion)) + facet_wrap(~type) + geom_hline(yintercept = 20.5) + geom_vline(xintercept = 20.5) + ggtitle('Proportion of Samples with Records Linked')"
R"ggplot(df3, aes(x = col, y = row)) + geom_tile(aes(fill = proportion)) + geom_hline(yintercept = 20.5) + geom_vline(xintercept = 20.5) + scale_fill_gradient2() + ggtitle('Standard Proportion - Step Proportion')"
R"dev.off()"
