"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a grid of MatchMatricies
"""
function metropolis_hastings_twostep{G <: Integer, T <: AbstractFloat}(
    nsamples::Int64,
    niter1::Int64,
    niter2::Int64,
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
    outC = Array{MatchMatrix}(nsamples)
    outM = Array{eltype(M0)}(nsamples, length(M0))
    outU = Array{eltype(U0)}(nsamples, length(U0))

    for ii in 1:nsamples
        println("sample ", ii, " of ", nsamples)
        #=
        #Sample from posterior
        draw = StatsBase.sample(1000:niter1)
        BMS1 = BlockMatchMatrix(BM0.nrows, BM0.ncols)
        for (jj, (rr, cc)) in enumerate(zip(blockrows1, blockcols1))
            BMS1.blocks[rr, cc] = S1BMArray[draw, jj]
        end
        exrows1, excols1 = getmatches(BMS1)

        #Map posterior sample to initial BlockMatchMatrix
        keep = map(x -> !in(x, exrows1), rows0) .* map(x -> !in(x, excols1), cols0)
        CS20 = MatchMatrix(rows0[keep], cols0[keep], BM0.nrow, BM0.ncol)
        BMS20 = BlockMatchMatrix(BM0.nrows, BM0.ncols, CS20)
        for (jj, (rr, cc)) in enumerate(zip(blockrows1, blockcols1))
            BMS20.blocks[rr, cc] = S1BMArray[draw, jj]
        end

        
        #MCMC for conditonal posterior
        S2BMArray, S2MArray, S2UArray, chgBM, chgM, chgU =
            metropolis_hastings_mixing(niter2, data, blockrows2, blockcols2, exrows1,
                                       excols1, BMS20, M1, U1, logpdfBM, logpdfM2,
                                       logpdfU, loglikelihood_datatable,
                                       transitionBM2, transitionM, transitionU,
                                       nBM = nBM, nM = nM, nU = nU)

        #Convert to standard MatchMatrix and save result
        for (jj, (rr, cc)) in enumerate(zip(blockrows2, blockcols2))
            BMS1.blocks[rr, cc] = S2BMArray[end, jj]
        end
        
        outC[ii] = MatchMatrix(getmatches(BMS1)..., BM0.nrow, BM0.ncol)
        outM[ii, :] = S2MArray[end, :]
        outU[ii, :] = S2UArray[end, :]
        =#

        outC[ii], outM[ii, :], outU[ii, :] = metropolis_hastings_conditional_sample(
            niter2,
            minidx,
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
    return outC, outM, outU
end

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
    BM0::BlockMatchMatrix{Int64},
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

    #Sample possible indcies
    draw = StatsBase.sample(minidx:size(blockArray, 1))

    #Transfer sampled entries to starting array
    BM = BlockMatchMatrix(BM0.nrows, BM0.ncols)
    for (jj, (rr, cc)) in enumerate(zip(condBlockrows, condBlockcols))
        BM.blocks[rr, cc] = blockArray[draw, jj]
    end
    exrows, excols = getmatches(BM)

    #Map posterior sample to initial BlockMatchMatrix
    rows0, cols0 = getmatches(BM0)
    #map(x -> !in(x, exrows), rows0) .* map(x -> !in(x, excols), cols0)
    keep = ![in(rr, exrows) || in(cc, excols) for (rr, cc) in zip(exrows, excols)]
    
    #Set values based on draw
    add_match!(BM, rows0[keep], cols0[keep])
    
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
end
