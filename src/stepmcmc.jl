"""
Draw a sample from Array{MatchMatrix, 2} and run conditional MCMC forward returning the last value
"""
function metropolis_hastings_conditional_sample{G <: Integer, T <: AbstractFloat}(
    minidx::Int64,
    niter::Int64,
    data::BitArray{3},
    condBlockrows::Array{G, 1},
    condBlockcols::Array{G, 1},
    nextBlockrows::Array{G, 1},
    nextBlockcols::Array{G, 1},
    blockArray::Array{MatchMatrix{G}, 2},
    BM0::BlockMatchMatrix{G},
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfBM::Function,
    logpdfM::Function,
    logpdfU::Function,
    loglikelihood::Function,
    transitionBM::Function,
    transitionM::Function,
    transitionU::Function;
    nBM::Int64 = 1,
    nM::Int64 = 1,
    nU::Int64 = 1)

    #Sample possible indicies
    draw = StatsBase.sample(minidx:size(blockArray, 1))

    #Transfer sampled entries to starting array
    BM = BlockMatchMatrix(BM0.nrows, BM0.ncols)
    for (jj, (rr, cc)) in enumerate(zip(condBlockrows, condBlockcols))
        BM.blocks[rr, cc] = blockArray[draw, jj]
    end
    exrows, excols = getmatches(BM)

    #Map posterior sample to initial BlockMatchMatrix and set remaining linkages
    rows0, cols0 = getmatches(nextBlockrows, nextBlockcols, BM0)
    keep = ![in(rr, exrows) || in(cc, excols) for (rr, cc) in zip(rows0, cols0)]
    add_match!(BM, rows0[keep], cols0[keep])    

    #Check that
    #mr, mc = getmatches(condBlockrows, condBlockrows, BM)
    #if !(issubset(mr, exrows) && issubset(exrows, mr)) || !(issubset(mc, excols) && issubset(excols, mc))
    #    println("exrows: ", exrows)
    #    println("excols: ", excols)
    #    println("nrows: ", BM.nrows)
    #    println("ncols: ", BM.ncols)
    #    println(BM)
    #    error("matches added to conditional blocks in initialization")
    #end
    
    
    #Generate Conditional Sample    
    outBM, outM, outU =
        metropolis_hastings_sample(niter,
                                   data,
                                   nextBlockrows,
                                   nextBlockcols,
                                   exrows,
                                   excols,
                                   BM,
                                   M0,
                                   U0,
                                   logpdfBM,
                                   logpdfM,
                                   logpdfU,
                                   loglikelihood_datatable,
                                   transitionBM,
                                   transitionM,
                                   transitionU,
                                   nBM = nBM, nM = nM, nU = nU)
    outC = MatchMatrix(getmatches(outBM)..., outBM.nrow, outBM.ncol)
    return outC, outM, outU
end

function metropolis_hastings_conditional_sample{G <: Integer, T <: AbstractFloat}(
    minidx::Int64,
    niter::Int64,
    data::BitArray{3},
    condBlockrows::Array{G, 1},
    condBlockcols::Array{G, 1},
    nextBlockrows::Array{G, 1},
    nextBlockcols::Array{G, 1},
    blockArray::Array{MatchMatrix{G}, 2},
    BM0::BlockMatchMatrix{G},
    perm0::Array{G, 1},
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfBM::Function,
    logpdfPerm::Function,
    logpdfM::Function,
    logpdfU::Function,
    loglikelihood::Function,
    transitionBM::Function,
    transitionPerm::Function,
    transitionM::Function,
    transitionU::Function;
    nBM::Int64 = 1,
    nPerm::Int64 = 1,
    nM::Int64 = 1,
    nU::Int64 = 1)

    println("MCMC Setup")
    #Sample possible indcies
    draw = StatsBase.sample(minidx:size(blockArray, 1))

    #Transfer sampled entries to starting array
    BM = BlockMatchMatrix(BM0.nrows, BM0.ncols)
    for (jj, (rr, cc)) in enumerate(zip(condBlockrows, condBlockcols))
        BM.blocks[rr, cc] = blockArray[draw, jj]
    end
    exrows, excols = getmatches(BM)

    #Map posterior sample to initial BlockMatchMatrix and set remaining linkages
    rows0, cols0 = getmatches(nextBlockrows, nextBlockcols, BM0)
    keep = ![in(rr, exrows) || in(cc, excols) for (rr, cc) in zip(rows0, cols0)]
    add_match!(BM, rows0[keep], cols0[keep])    

    println("Running MCMC")
    #Generate Conditional Sample
    outBM, outPerm, outM, outU =
        metropolis_hastings_permutation_sample(niter,
                                               data,
                                               nextBlockrows,
                                               nextBlockcols,
                                               exrows,
                                               excols,
                                               BM,
                                               perm0,
                                               M0,
                                               U0,
                                               logpdfBM,
                                               logpdfPerm,
                                               logpdfM,
                                               logpdfU,
                                               loglikelihood_datatable,
                                               transitionBM,
                                               transitionPerm,
                                               transitionM,
                                               transitionU,
                                               nBM = nBM, nPerm = nPerm, nM = nM, nU = nU)
    outC = MatchMatrix(getmatches(outBM)..., outBM.nrow, outBM.ncol)
    return outC, outPerm, outM, outU
end


"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a grid of MatchMatricies
"""
function metropolis_hastings_twostep{G <: Integer, T <: AbstractFloat}(
    nsamples::Int64,
    niter1::Int64,
    niter2::Int64,
    minidx::Int64,
    data::BitArray{3},
    blockrows1::Array{G, 1},
    blockcols1::Array{G, 1},
    blockrows2::Array{G, 1},
    blockcols2::Array{G, 1},
    BM0::BlockMatchMatrix{G},
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
    nU::Int64 = 1)

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

    for ii in 1:nsamples
        println("sample ", ii, " of ", nsamples)

        outC[ii], outM[ii, :], outU[ii, :] = metropolis_hastings_conditional_sample(
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
    return outC, outM, outU, S1BMArray, S1MArray, S1UArray
end


function metropolis_hastings_twostep{G <: Integer, T <: AbstractFloat}(
    nsamples::Int64,
    niter1::Int64,
    niter2::Int64,
    minidx::Int64,
    data::BitArray{3},
    blockrows1::Array{G, 1},
    blockcols1::Array{G, 1},
    blockrows2::Array{G, 1},
    blockcols2::Array{G, 1},
    BM0::BlockMatchMatrix{G},
    perm0::Array{G, 1},
    perm1::Array{G, 1}, #should be same length as blockrow2
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfBM::Function,
    logpdfPerm::Function,
    logpdfM::Function,
    logpdfU::Function,
    loglikelihood::Function,
    transitionBM::Function,
    transitionPerm::Function,
    transitionM::Function,
    transitionU::Function,
    θ::Float64,
    p::Float64;
    nBM::Int64 = 1,
    nPerm::Int64 = 1,
    nM::Int64 = 1,
    nU::Int64 = 1)

    rows0, cols0 = getmatches(BM0)

    BM1 = BlockMatchMatrix(BM0.nrows, BM0.ncols)
    for (ii, jj) in zip(blockrows1, blockcols1)
        BM1.blocks[ii, jj] = BM0.blocks[ii, jj]
    end
    
    println("MCMC For Step 1")
    S1BMArray, S1PermArray, S1MArray, S1UArray, chgBM, chgPerm, chgM, chgU =
        metropolis_hastings_permutation(niter1, data, blockrows1, blockcols1, BM1, perm0, M0, U0,
                                        logpdfBM, logpdfPerm, logpdfM, logpdfU, loglikelihood,
                                        transitionBM, transitionPerm, transitionM, transitionU,
                                        nBM = nBM, nM = nM, nU = nU)
    
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
    outPerm = Array{G}(nsamples, length(perm1))
    outM = Array{T}(nsamples, length(M0))
    outU = Array{T}(nsamples, length(U0))

    for ii in 1:nsamples
        println("sample ", ii, " of ", nsamples)

        outC[ii], outPerm[ii, :],  outM[ii, :], outU[ii, :] = metropolis_hastings_conditional_sample(
            minidx,
            niter2,
            data,
            blockrows1,
            blockcols1,
            blockrows2,
            blockcols2,
            S1BMArray,
            BM0,
            perm1,
            M1,
            U1,
            logpdfBM,
            logpdfPerm,
            logpdfM2,
            logpdfU,
            loglikelihood,
            transitionBM2,
            transitionPerm,
            transitionM,
            transitionU,
            nBM = nBM,
            nPerm = nPerm,
            nM = nM,
            nU = nU)
    end
    return outC, outPerm, outM, outU, S1BMArray, S1PermArray, S1MArray, S1UArray
end
