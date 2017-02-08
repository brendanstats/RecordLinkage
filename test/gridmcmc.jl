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

#Log prior density functions for standard algorithm
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
niter = 10000
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

R"par(mfrow = c(2,3))
plot(density($MArray[,1]), xlim = c(0, 1), main = 'M1')
abline(v = $pM[1], col = 2)
plot(density($MArray[,2]), xlim = c(0, 1), main = 'M2')
abline(v = $pM[2], col = 2)
plot(density($MArray[,3]), xlim = c(0, 1), main = 'M3')
abline(v = $pM[3], col = 2)
plot(density($UArray[,1]), xlim = c(0, 1), main = 'U1')
abline(v = $pU[1], col = 2)
plot(density($UArray[,2]), xlim = c(0, 1), main = 'U2')
abline(v = $pU[2], col = 2)
plot(density($UArray[,3]), xlim = c(0, 1), main = 'U3')
abline(v = $pU[3], col = 2)
par(mfrow = c(1,1))"

M1kde = unitkde_slow(vec(MArray[:,1]), 512, .01)
M1kdetilt = unitkde_tilted(vec(MArray[:,1]), 512, .01, .1)
x2 = Distributions.rand(M1kdetilt, 10000)

R"plot(density($MArray[,1]), xlim = c(0, 1), main = 'M1')
lines($(M1kde.x), $(M1kde.y), xlim = c(0, 1), lty = 2)
lines($(M1kdetilt.x), $(M1kdetilt.y), xlim = c(0, 1), lty = 3)
lines(density($x2), xlim = c(0, 1), lty = 3, col = 2)
legend('topleft', legend = c('R KDE', 'Custom KDE', 'Tilted KDE'))"

#=
Blocking Case
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

#Run algorithm
srand(53177)

@time S1GMArray, S1MArray, S1UArray, chgGM, chgM, chgU = metropolis_hastings_mixing(niter,
                                                                                    data,
                                                                                    [1, 2],
                                                                                    [1, 2],
                                                                                    GM0,
                                                                                    M0,
                                                                                    U0,
                                                                                    lpGM,
                                                                                    lpM,
                                                                                    lpU,
                                                                                    loglikelihood_datatable,
                                                                                    transGM,
                                                                                    transMGrid,
                                                                                    transUGrid)
mean(chgM)
mean(chgU)

R"par(mfrow = c(2,3))
plot(density($S1MArray[,1]), xlim = c(0, 1), main = 'M1')
abline(v = $pM[1], col = 2)
plot(density($S1MArray[,2]), xlim = c(0, 1), main = 'M2')
abline(v = $pM[2], col = 2)
plot(density($S1MArray[,3]), xlim = c(0, 1), main = 'M3')
abline(v = $pM[3], col = 2)
plot(density($S1UArray[,1]), xlim = c(0, 1), main = 'U1')
abline(v = $pU[1], col = 2)
plot(density($S1UArray[,2]), xlim = c(0, 1), main = 'U2')
abline(v = $pU[2], col = 2)
plot(density($S1UArray[,3]), xlim = c(0, 1), main = 'U3')
abline(v = $pU[3], col = 2)
par(mfrow = c(1,1))"

#Create new priors based on posteriors from previous step
kdetiltS1M1 = unitkde_tilted(vec(S1MArray[:,1]), 512, .01, .1)
kdetiltS1M2 = unitkde_tilted(vec(S1MArray[:,2]), 512, .01, .1)
kdetiltS1M3 = unitkde_tilted(vec(S1MArray[:,3]), 512, .01, .1)
function lpMS2{T <: AbstractFloat}(γM::Array{T, 1})
    return log(unitkde_interpolate(γM[1], kdetiltS1M1)) + log(unitkde_interpolate(γM[2], kdetiltS1M2)) + log(unitkde_interpolate(γM[3], kdetiltS1M3))
end

kdetiltS1U1 = unitkde_slow(vec(S1UArray[:,1]), 512, .01)
kdetiltS1U2 = unitkde_slow(vec(S1UArray[:,2]), 512, .01)
kdetiltS1U3 = unitkde_slow(vec(S1UArray[:,3]), 512, .01)
function lpUS2{T <: AbstractFloat}(γU::Array{T, 1})
    return log(unitkde_interpolate(γU[1], kdetiltS1U1)) + log(unitkde_interpolate(γU[2], kdetiltS1U2)) + log(unitkde_interpolate(γU[3], kdetiltS1U3))
end

function transGMS2{G <: Integer}(grows::Array{G, 1}, gcols::Array{G, 1}, exrows::Array{G, 1}, excols::Array{G, 1}, GM::GridMatchMatrix)
    return move_gridmatchmatrix_exclude(grows, gcols, exrows, excols, GM, 0.5)
end

#Sample from blocked posterior
draw = StatsBase.sample(1000:niter)
GMS1 = GridMatchMatrix([20,20], [20,20])
for (ii, (rr, cc)) in enumerate(zip([1, 2], [1, 2]))
    GMS1.grid[rr, cc] = S1GMArray[draw, ii]
end

mrows, mcols = getmatches(GMS1)
keep = map(x -> !in(x, mrows), rows) .* map(x -> !in(x, mcols), cols)
CS20 = MatchMatrix(rows[keep], cols[keep], 40, 40)
GMS20 = GridMatchMatrix([20,20], [20,20], CS20)
for (ii, (rr, cc)) in enumerate(zip([1, 2], [1, 2]))
    GMS20.grid[rr, cc] = S1GMArray[draw, ii]
end

srand(30271)

@time S2GMArray, S2MArray, S2UArray, chgGM, chgM, chgU = metropolis_hastings_mixing(niter,
                                                                                    data,
                                                                                    [2, 1],
                                                                                    [1, 2],
                                                                                    mrows,
                                                                                    mcols,
                                                                                    GMS20,
                                                                                    M0,
                                                                                    U0,
                                                                                    lpGM,
                                                                                    lpM,
                                                                                    lpU,
                                                                                    loglikelihood_datatable,
                                                                                    transGMS2,
                                                                                    transMGrid,
                                                                                    transUGrid)
R"par(mfrow = c(2,3))
plot(density($S2MArray[,1]), xlim = c(0, 1), main = 'M1')
abline(v = $pM[1], col = 2)
plot(density($S2MArray[,2]), xlim = c(0, 1), main = 'M2')
abline(v = $pM[2], col = 2)
plot(density($S2MArray[,3]), xlim = c(0, 1), main = 'M3')
abline(v = $pM[3], col = 2)
plot(density($S2UArray[,1]), xlim = c(0, 1), main = 'U1')
abline(v = $pU[1], col = 2)
plot(density($S2UArray[,2]), xlim = c(0, 1), main = 'U2')
abline(v = $pU[2], col = 2)
plot(density($S2UArray[,3]), xlim = c(0, 1), main = 'U3')
abline(v = $pU[3], col = 2)
par(mfrow = c(1,1))"
