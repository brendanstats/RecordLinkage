"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a grid of MatchMatricies
"""
function metropolis_hastings_twostep{G <: Integer, T <: AbstractFloat}(
    nsamples::Int64,
    niter1::Int64,
    niter2::Int64,
    data::BitArray{3},
    grows1::Array{G, 1},
    gcols1::Array{G, 1},
    grows2::Array{G, 1},
    gcols2::Array{G, 1},
    GM0::GridMatchMatrix,
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfGM::Function,
    logpdfM::Function,
    logpdfU::Function,
    loglikelihood::Function,
    transitionGM::Function,
    transitionM::Function,
    transitionU::Function,
    θ::Float64,
    p::Float64;
    nGM::Int64 = 1,
    nM::Int64 = 1,
    nU::Int64 = 1)

    rows0, cols0 = getmatches(GM0)
    
    println("MCMC For Step 1")
    S1GMArray, S1MArray, S1UArray, chgGM, chgM, chgU =
        metropolis_hastings_mixing(niter1, data, grows1, gcols1, GM0, M0, U0,
                                   logpdfGM, logpdfM, logpdfU, loglikelihood,
                                   transitionGM, transitionM, transitionU, nGM = nGM,
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
    function transitionGM2{G <: Integer}(grows::Array{G, 1}, gcols::Array{G, 1}, exrows::Array{G, 1}, excols::Array{G, 1}, GM::GridMatchMatrix)
        return move_gridmatchmatrix_exclude(grows, gcols, exrows, excols, GM, 0.5)
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
        GMS1 = GridMatchMatrix(GM0.nrows, GM0.ncols)
        for (jj, (rr, cc)) in enumerate(zip(grows1, gcols1))
            GMS1.grid[rr, cc] = S1GMArray[draw, jj]
        end
        exrows1, excols1 = getmatches(GMS1)

        #Map posterior sample to initial GridMatchMatrix
        keep = map(x -> !in(x, exrows1), rows0) .* map(x -> !in(x, excols1), cols0)
        CS20 = MatchMatrix(rows0[keep], cols0[keep], GM0.nrow, GM0.ncol)
        GMS20 = GridMatchMatrix(GM0.nrows, GM0.ncols, CS20)
        for (jj, (rr, cc)) in enumerate(zip(grows1, gcols1))
            GMS20.grid[rr, cc] = S1GMArray[draw, jj]
        end

        
        #MCMC for conditonal posterior
        S2GMArray, S2MArray, S2UArray, chgGM, chgM, chgU =
            metropolis_hastings_mixing(niter2, data, grows2, gcols2, exrows1,
                                       excols1, GMS20, M1, U1, logpdfGM, logpdfM2,
                                       logpdfU, loglikelihood_datatable,
                                       transitionGM2, transitionM, transitionU,
                                       nGM = nGM, nM = nM, nU = nU)

        #Convert to standard MatchMatrix and save result
        for (jj, (rr, cc)) in enumerate(zip(grows2, gcols2))
            GMS1.grid[rr, cc] = S2GMArray[end, jj]
        end
        
        outC[ii] = MatchMatrix(getmatches(GMS1)..., GM0.nrow, GM0.ncol)
        outM[ii, :] = S2MArray[end, :]
        outU[ii, :] = S2UArray[end, :]
        =#

        outC[ii], outM[ii, :], outU[ii, :] = metropolis_hastings_conditional_sample(
            niter2,
            minidx,
            data,
            grows1,
            gcols1,
            grows2,
            gcols2,
            S1GMArray,
            GM0,
            M1,
            U1,
            logpdfGM,
            logpdfM2,
            logpdfU,
            loglikelihood,
            transitionGM2,
            transitionM,
            transitionU,
            nGM = nGM,
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
    condGRows::Array{G, 1},
    condGCols::Array{G, 1},
    nextGRows::Array{G, 1},
    nextGCols::Array{G, 1},
    blockArray::Array{MatchMatrix{G}, 2},
    GM0::GridMatchMatrix{Int64},
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfGM::Function,
    logpdfM::Function,
    logpdfU::Function,
    loglikelihood::Function,
    transitionGM::Function,
    transitionM::Function,
    transitionU::Function;
    nGM::Int64 = 1,
    nM::Int64 = 1,
    nU::Int64 = 1)

    #Sample possible indcies
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
    
    outGM, outM, outU =
        metropolis_hastings_sample(niter,
                                   data,
                                   nextGRows,
                                   nextGCols,
                                   exrows,
                                   excols,
                                   GM,
                                   M0,
                                   U0,
                                   logpdfGM,
                                   logpdfM,
                                   logpdfU,
                                   loglikelihood_datatable,
                                   transitionGM,
                                   transitionM,
                                   transitionU,
                                   nGM = nGM, nM = nM, nU = nU)
    outC = MatchMatrix(getmatches(outGM)..., outGM.nrow, outGM.ncol)
end
