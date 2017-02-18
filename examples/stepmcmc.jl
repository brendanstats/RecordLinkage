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
function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = Distributions.Beta(5, 2)
    return sum(Distributions.logpdf(d, γM))
end

function lpU{T <: AbstractFloat}(γU::Array{T, 1})
    d = Distributions.Beta(2, 20)
    return sum(Distributions.logpdf(d, γU))
end

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


nsamples = 100
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

write_matchmatrix("match_results.txt", outC)
write_probs("prob_results.txt", outM, outU)

nmatches, matchcounts = totalmatches(outC)
plotdata = gridtoarray(matchcounts)

R"library(ggplot2)"

@rput plotdata
R"df <- as.data.frame(plotdata)"
R"names(df) <- c('row', 'col', 'count')"

#R"ggplot(df, aes(x = col, y = row)) + geom_raster(aes(fill = count))"
R"ggplot(df, aes(x = col, y = row)) + geom_tile(aes(fill = count)) +
geom_text(aes(label = count), size = 2) +
geom_point(aes(x = x, y = y), alpha = 0.4, color = 'red',
data = data.frame(x = $(C.cols), y = $(C.rows)))"

R"table(df$count)"
R"hist($nmatches)"
