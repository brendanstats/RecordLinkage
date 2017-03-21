"""
Function to draw a sample given the results of an MCMC algorithm on a subset of the linkage matrix
"""
function metropolis_hastings_ptwostep{G <: Integer, T <: AbstractFloat}(
    nsamples::Int64,
    niter1::Int64,
    niter2::Int64,
    minidx::Int64,
    data::BitArray{3},
    blockrows1::Array{G, 1},
    blockcols1::Array{G, 1},
    blockrows2::Array{G, 1},
    blockcols2::Array{G, 1},
    BM0::BlockMatchMatrix,
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfBM::Function,
    logpdfM::Function,
    logpdfU::Function,
    loglikelihood::Function,
    transitionBM::Function,
    transitionM::Function,
    transitionU::Function,
    θ::Float64,
    p::Float64;
    nBM::Int64 = 1,
    nM::Int64 = 1,
    nU::Int64 = 1,
    np::Int64 = nprocs())

    rows0, cols0 = getmatches(BM0)
    
    println("MCMC For Step 1")
    S1BMArray, S1MArray, S1UArray, chgBM, chgM, chgU =
        metropolis_hastings_mixing(niter1, data, blockrows1, blockcols1, BM0, M0, U0,
                                   logpdfBM, logpdfM, logpdfU, loglikelihood,
                                   transitionBM, transitionM, transitionU, nBM = nBM,
                                   nM = nM, nU = nU)

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
    function transitionBM2{G <: Integer}(blockrows::Array{G, 1}, blockcols::Array{G, 1}, exrows::Array{G, 1}, excols::Array{G, 1}, BM::BlockMatchMatrix)
        return move_blockmatchmatrix_exclude(blockrows, blockcols, exrows, excols, BM, 0.5)
    end
    
    #Allocate Arrays for results
    outC = Array{MatchMatrix{G}}(nsamples)
    outM = Array{T}(nsamples, length(M0))
    outU = Array{T}(nsamples, length(U0))

    @sync begin
        for pr = 1:np
            if pr != myid() || np == 1
                @async begin
                    while true
                        idx += 1
                        if idx > nsamples
                            break
                        end
                        outC[idx], outM[idx, :], outU[idx, :] = remotecall_fetch(metropolis_hastings_conditional_sample, pr,
                                                                                 minidx,
                                                                                 niter2,
                                                                                 data,
                                                                                 blockrows1,
                                                                                 blockcols1,
                                                                                 blockrows2,
                                                                                 blockcols2,
                                                                                 S1BMArray,
                                                                                 BM0,
                                                                                 M1,
                                                                                 U1,
                                                                                 logpdfBM,
                                                                                 logpdfM2,
                                                                                 logpdfU,
                                                                                 loglikelihood,
                                                                                 transitionBM2,
                                                                                 transitionM,
                                                                                 transitionU,
                                                                                 nBM = nBM,
                                                                                 nM = nM,
                                                                                 nU = nU)
                    end
                end
            end
        end
    end
    return outC, outM, outU
end
