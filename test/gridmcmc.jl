push!(LOAD_PATH, "/Users/Brendan/Google Drive/2016_S2_Fall/Record Linkage/code/src")
using SequentialRecordLinkage

pM = [0.8, 0.9, 0.68]
pU = [0.15, 0.08, .45]
srand(68259)
C = rand(UniformSingleLinkage(40, 40, 27))
data = simulate_singlelinkage_binary(C, pM, pU)
database = [ones(Int8, 20, 20) zeros(Int8, 20, 20); zeros(Int8, 20, 20) ones(Int8, 20, 20)]
nones = countones(data)
gridarray = gridtoarray(nones)

rows = gridarray[gridarray[:, 3] .== 3, 1]
cols = gridarray[gridarray[:, 3] .== 3, 2]
deleteat!(rows, (2, 10, 11, 12, 14, 21, 25, 26))
deleteat!(cols, (2, 10, 11, 12, 14, 21, 25, 26))

C0 = MatchMatrix(rows, cols, 40, 40)
M0 = [0.99, 0.99, 0.99]
U0 = vec((sum(data, 2:3) .- length(rows)) ./ (40 * 40 - length(rows)))
GM0 = GridMatchMatrix([20,20], [20,20], C0)

#Log prior densities
#logpdfC::Function
function lpC(C::MatchMatrix)
    return -length(C.rows) * 0.6
end
#M probabilities logpdfM::Function
function lpM{T <: AbstractFloat}(γM::Array{T, 1})
    d = Distributions.Beta(5, 2)
    return sum(Distributions.logpdf(d, γM))
end

#U probabilities logpdfU::Function
function lpU{T <: AbstractFloat}(γU::Array{T, 1})
    d = Distributions.Beta(2, 20)
    return sum(Distributions.logpdf(d, γU))
end

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

#Transition Functions
function transC(C::MatchMatrix)
    return move_matchmatrix(C, 0.5)
end

#Transition Ratio
function transC_ratio(M1::MatchMatrix, M2::MatchMatrix)
    return ratio_pmove(M1, M2, 0.5)
end

niter = 1000
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

#Blocking Case
function lpGM(grows::Array{Int64, 1}, gcols::Array{Int64, 1}, GM::GridMatchMatrix)
    θ = 0.6
    l = 0
    for (ii, jj) in zip(grows, gcols)
        l += length(GM.grid[ii, jj].rows)
    end
    return -θ * l
end

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

GMArray, MArray, UArray, chgGM, chgM, chgU = metropolis_hastings_mixing(100000,
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
