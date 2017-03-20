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

function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = Distributions.Beta(5, 2)
    return sum(Distributions.logpdf(d, γM))
end

function lpU{T <: AbstractFloat}(γU::Array{T, 1})
    d = Distributions.Beta(2, 20)
    return sum(Distributions.logpdf(d, γU))
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

nsamples = 10
niter1 = 1000
niter2 = 1000

grows1 = [1, 2]
gcols1 = [1, 2]
grows2 = [1, 2]
gcols2 = [2, 1]
θ = 0.1
p = 0.1

srand(29171)

#TEST ON INTERNAL FUNCTIONS FIRST
println("MCMC For Step 1")

S1GMArray, S1MArray, S1UArray, chgGM, chgM, chgU =
    metropolis_hastings_mixing(niter1, data[:, :, 2:4], grows1, gcols1, GM0, M0, U0,
                               lpGM, lpM, lpU, loglikelihood_datatable,
                               transGM, transMGrid, transUGrid)

println("Processing Step 1 Results...")
#Generate new match prior distributions from posteriors
ukdeM = map(x -> unitkde_tilted(vec(S1MArray[:,x]), 512, .01, θ), 1:size(S1MArray, 2))
M1 = Distributions.mode.(ukdeM)
ukdeU = map(x -> unitkde_slow(vec(S1UArray[:,x]), 512, .01), 1:size(S1UArray, 2))
U1 = Distributions.mode.(ukdeU)

dA = beta_mode.(M1, 3.0)
mixtureM = [UnitKDEMixture(ukde, d, p) for (ukde, d) in zip(ukdeM, dA)]
function logpdfM2{T <: AbstractFloat}(γM::Array{T, 1})
    return sum([Distributions.logpdf(d, pi) for (d, pi) in zip(mixtureM, γM)])
end
function transitionGM2{G <: Integer}(grows::Array{G, 1}, gcols::Array{G, 1}, exrows::Array{G, 1}, excols::Array{G, 1}, GM::GridMatchMatrix)
    return move_gridmatchmatrix_exclude(grows, gcols, exrows, excols, GM, 0.5)
end


#check internals of conditional_sample
blockArray = S1GMArray

draw = StatsBase.sample(minidx:size(blockArray, 1))

#Transfer sampled entries to starting array
GM = GridMatchMatrix(GM0.nrows, GM0.ncols)
for (jj, (rr, cc)) in enumerate(zip(condGRows, condGCols))
    GM.grid[rr, cc] = blockArray[draw, jj]
end
exrows, excols = getmatches(GM)

#Map posterior sample to initial GridMatchMatrix
rows0, cols0 = getmatches(GM0)
#map(x -> !in(x, exrows), rows0) .* map(x -> !in(x, excols), cols0)
keep = ![in(rr, exrows) || in(cc, excols) for (rr, cc) in zip(exrows, excols)]

#Set values based on draw
add_match!(GM, rows0[keep], cols0[keep])

#Allocate Arrays for results
outC = Array{MatchMatrix}(nsamples)
outM = Array{eltype(M0)}(nsamples, length(M0))
outU = Array{eltype(U0)}(nsamples, length(U0))

ii = 1

outC[ii], outM[ii, :], outU[ii, :] = metropolis_hastings_conditional_sample(
    niter2,
    100,
    data,
    grows1,
    gcols1,
    grows2,
    gcols2,
    S1GMArray,
    GM0,
    M1,
    U1,
    lpGM,
    logpdfM2,
    lpU,
    loglikelihood_datatable,
    transitionGM2,
    transMGrid,
    transUGrid)




#Test wrapper function
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

    rows0, cols0 = getmatches(GM0)
