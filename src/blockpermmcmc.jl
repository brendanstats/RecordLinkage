"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a grid of MatchMatricies
"""
function metropolis_hastings_permutation{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::BitArray{3},
    blockrows::Array{G, 1},
    blockcols::Array{G, 1},
    BM0::BlockMatchMatrix{G},
    perm0::Array{G, 1},
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfBM::Function, #now depends on perm, should contain matrix_logprobability_single_linkage
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

    #Check Lengths match
    if length(blockrows) != length(blockcols)
        error("length of blockrows and block columns must match")
    end
    if length(blockrows) != length(perm0)
        error("length of blockrows and permutation must match")
    end
    
    #MCMC Chains
    BMArray = Array{MatchMatrix{G}}(niter, length(blockrows))
    PermArray = Array{G}(niter, length(blockrows))
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    #Track transitions
    transBM = falses(niter)
    transPerm = falses(niter)
    transM = falses(niter)
    transU = falses(niter)

    #Initial States
    currBM = copy(BM0)
    currTable = data2table(data, blockrows, blockcols, currBM)
    currBlockNLinks = getblocknlinks(currBM)
    currPerm = perm0
    currBRows = blockrows[currPerm]
    currBCols = blockcols[currPerm]
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for BM
        for gg in 1:nBM
            propBM, ratioBM = transitionBM(blockrows, blockcols, currBM)
            propTable = data2table(data, blockrows, blockcols, propBM)
            propBlockNLinks = getblocknlinks(propBM)
            
            #compute a1
            a1 = exp(logpdfBM(propBlockNLinks, currBRows, currBCols) + loglikelihood(propTable, currM, currU) - logpdfBM(currBlockNLinks, currBRows, currBCols) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioBM

            #accept / reject move
            if rand() < a1 * a2
                currBM = propBM
                currTable = propTable
                currBlockNLinks = propBlockNLinks
                transBM[ii] = true
            end
        end

        #Inner iteration for permutation
        for pp in nPerm
            propPerm, ratioPerm, = transitionPerm(currPerm)
            propBRows = blockrows[propPerm]
            propBCols = blockcols[propPerm]
            
            #compute a1
            a1 = exp(logpdfBM(currBlockNLinks, propBRows, propBCols) + logpdfPerm(propPerm) - logpdfBM(currBlockNLinks, currBRows, currBCols) - logpdfPerm(currPerm))

            #compute a2
            a2 = ratioPerm

            #accept / reject move
            if rand() < a1 * a2
                currPerm = propPerm
                currBRows = propBRows
                currBCols = propBCols
                transPerm[ii] = true
            end            
        end
        
        #Inner iteration for M
        for mm in nM
            propM, ratioM = transitionM(currM)

            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioM

            #accept / reject move
            if rand() < a1 * a2
                currM = propM
                transM[ii] = true
            end
        end

        #Inner iteration for U
        for uu in nU
            propU, ratioU = transitionU(currU)

            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioU

            #accept / reject move
            if rand() < a1 * a2
                currU = propU
                transU[ii] = true
            end
        end

        #Add states to chain
        for (jj, (rr, cc)) in enumerate(zip(blockrows, blockcols))
            BMArray[ii, jj] = currBM.blocks[rr, cc]
        end
        PermArray[ii, :] = currPerm
        MArray[ii, :] = currM
        UArray[ii, :] = currU
        
    end
    return BMArray, PermArray, MArray, UArray, transBM, transPerm, transM, transU
end

"""
Allow rows to be excluded
"""
function metropolis_hastings_permutation{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::BitArray{3},
    blockrows::Array{G, 1},
    blockcols::Array{G, 1},
    exrows::Array{G, 1}, #think about how to incorporate into logpdfBM through logprobability_blocknlinks_single_linkage through nrowsEff and ncolsEff
    excols::Array{G, 1},
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
    
    #Check Lengths match
    if length(blockrows) != length(blockcols)
        error("length of blockrows and block columns must match")
    end
    if length(blockrows) != length(perm0)
        error("length of blockrows and permutation must match")
    end

    #Determine remaining block sizes
    nrowsRemain = getnrowsremaining(BM0, exrows)
    ncolsRemain = getncolsremaining(BM0, excols)
    
    #MCMC Chains
    BMArray = Array{MatchMatrix{G}}(niter, length(blockrows))
    PermArray = Array{G}(niter, length(blockrows))
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    #Track transitions
    transBM = falses(niter)
    transPerm = falses(niter)
    transM = falses(niter)
    transU = falses(niter)

    #Initial States
    currBM = copy(BM0)
    currTable = data2table(data, blockrows, blockcols, currBM)
    currBlockNLinks = getblocknlinks(currBM)
    currPerm = perm0
    currBRows = blockrows[currPerm]
    currBCols = blockcols[currPerm]
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for BM
        for gg in 1:nBM
            propBM, ratioBM = transitionBM(blockrows, blockcols, exrows, excols, currBM)
            propTable = data2table(data, blockrows, blockcols, propBM)
            propBlockNLinks = getblocknlinks(propBM)
            
            #compute a1
            a1 = exp(logpdfBM(propBlockNLinks, currBRows, currBCols, nrowsRemain, ncolsRemain) + loglikelihood(propTable, currM, currU) - logpdfBM(currBlockNLinks, currBRows, currBCols, nrowsRemain, ncolsRemain) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioBM

            #accept / reject move
            if rand() < a1 * a2
                currBM = propBM
                currTable = propTable
                currBlockNLinks = propBlockNLinks
                transBM[ii] = true
            end
        end

        #Inner iteration for permutation
        for pp in nPerm
            propPerm, ratioPerm, = transitionPerm(currPerm)
            propBRows = blockrows[propPerm]
            propBCols = blockcols[propPerm]
            
            #compute a1
            a1 = exp(logpdfBM(currBlockNLinks, propBRows, propBCols, nrowsRemain, ncolsRemain) + logpdfPerm(propPerm) - logpdfBM(currBlockNLinks, currBRows, currBCols, nrowsRemain, ncolsRemain) - logpdfPerm(currPerm))

            #compute a2
            a2 = ratioPerm

            #accept / reject move
            if rand() < a1 * a2
                currPerm = propPerm
                currBRows = propBRows
                currBCols = propBCols
                transPerm[ii] = true
            end            
        end
        
        #Inner iteration for M
        for mm in nM
            propM, ratioM = transitionM(currM)

            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioM

            #accept / reject move
            if rand() < a1 * a2
                currM = propM
                transM[ii] = true
            end
        end

        #Inner iteration for U
        for uu in nU
            propU, ratioU = transitionU(currU)

            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioU

            #accept / reject move
            if rand() < a1 * a2
                currU = propU
                transU[ii] = true
            end
        end

        #Add states to chain
        for (jj, (rr, cc)) in enumerate(zip(blockrows, blockcols))
            BMArray[ii, jj] = currBM.blocks[rr, cc]
        end
        PermArray[ii, :] = currPerm
        MArray[ii, :] = currM
        UArray[ii, :] = currU
        
    end
    return BMArray, PermArray, MArray, UArray, transBM, transPerm, transM, transU
end

"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a grid of MatchMatricies
"""
function metropolis_hastings_permutation_sample{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::BitArray{3},
    blockrows::Array{G, 1},
    blockcols::Array{G, 1},
    BM0::BlockMatchMatrix{G},
    perm0::Array{G, 1},
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfBM::Function, #now depends on perm, should contain matrix_logprobability_single_linkage
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

    #Check Lengths match
    if length(blockrows) != length(blockcols)
        error("length of blockrows and block columns must match")
    end
    if length(blockrows) != length(perm0)
        error("length of blockrows and permutation must match")
    end
    
    #Initial States
    currBM = BM0
    currTable = data2table(data, blockrows, blockcols, currBM)
    currBlockNLinks = getblocknlinks(currBM)
    currPerm = perm0
    currBRows = blockrows[currPerm]
    currBCols = blockcols[currPerm]
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for BM
        for gg in 1:nBM
            propBM, ratioBM = transitionBM(blockrows, blockcols, currBM)
            propTable = data2table(data, blockrows, blockcols, propBM)
            propBlockNLinks = getblocknlinks(propBM)
            
            #compute a1
            a1 = exp(logpdfBM(propBlockNLinks, currBRows, currBCols) + loglikelihood(propTable, currM, currU) - logpdfBM(currBlockNLinks, currBRows, currBCols) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioBM

            #accept / reject move
            if rand() < a1 * a2
                currBM = propBM
                currTable = propTable
                currBlockNLinks = propBlockNLinks
            end
        end

        #Inner iteration for permutation
        for pp in nPerm
            propPerm, ratioPerm, = transitionPerm(currPerm)
            propBRows = blockrows[propPerm]
            propBCols = blockcols[propPerm]
            
            #compute a1
            a1 = exp(logpdfBM(currBlockNLinks, propBRows, propBCols) + logpdfPerm(propPerm) - logpdfBM(currBlockNLinks, currBRows, currBCols) - logpdfPerm(currPerm))

            #compute a2
            a2 = ratioPerm

            #accept / reject move
            if rand() < a1 * a2
                currPerm = propPerm
                currBRows = propBRows
                currBCols = propBCols
            end            
        end
        
        #Inner iteration for M
        for mm in nM
            propM, ratioM = transitionM(currM)

            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioM

            #accept / reject move
            if rand() < a1 * a2
                currM = propM
            end
        end

        #Inner iteration for U
        for uu in nU
            propU, ratioU = transitionU(currU)

            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioU

            #accept / reject move
            if rand() < a1 * a2
                currU = propU
            end
        end
 
    end
    return currBM, currPerm, currM, currU
end

"""
Allow rows to be excluded
"""
function metropolis_hastings_permutation_sample{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::BitArray{3},
    blockrows::Array{G, 1},
    blockcols::Array{G, 1},
    exrows::Array{G, 1}, #think about how to incorporate into logpdfBM through logprobability_blocknlinks_single_linkage through nrowsEff and ncolsEff
    excols::Array{G, 1},
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

    #Check Lengths match
    if length(blockrows) != length(blockcols)
        error("length of blockrows and block columns must match")
    end
    if length(blockrows) != length(perm0)
        error("length of blockrows and permutation must match")
    end

    #Determine remaining block sizes
    nrowsRemain = getnrowsremaining(BM0, exrows)
    ncolsRemain = getncolsremaining(BM0, excols)
    
    #Initial States
    currBM = BM0
    currTable = data2table(data, blockrows, blockcols, currBM)
    currBlockNLinks = getblocknlinks(currBM)
    currPerm = perm0
    currBRows = blockrows[currPerm]
    currBCols = blockcols[currPerm]
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        #println("Iteration ", ii)
        
        #Inner iteration for BM
        #println("Moving BM")
        for gg in 1:nBM
            propBM, ratioBM = transitionBM(blockrows, blockcols, exrows, excols, currBM)
            propTable = data2table(data, blockrows, blockcols, propBM)
            propBlockNLinks = getblocknlinks(propBM)
            
            #compute a1
            a1 = exp(logpdfBM(propBlockNLinks, currBRows, currBCols, nrowsRemain, ncolsRemain) + loglikelihood(propTable, currM, currU) - logpdfBM(currBlockNLinks, currBRows, currBCols, nrowsRemain, ncolsRemain) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioBM

            #accept / reject move
            if rand() < a1 * a2
                currBM = propBM
                currTable = propTable
                currBlockNLinks = propBlockNLinks
            end
        end

        #Inner iteration for permutation
        #println("Moving Permuation")
        for pp in nPerm
            propPerm, ratioPerm, = transitionPerm(currPerm)
            propBRows = blockrows[propPerm]
            propBCols = blockcols[propPerm]
            
            #compute a1
            a1 = exp(logpdfBM(currBlockNLinks, propBRows, propBCols, nrowsRemain, ncolsRemain) + logpdfPerm(propPerm) - logpdfBM(currBlockNLinks, currBRows, currBCols, nrowsRemain, ncolsRemain) - logpdfPerm(currPerm))

            #compute a2
            a2 = ratioPerm

            #accept / reject move
            if rand() < a1 * a2
                currPerm = propPerm
                currBRows = propBRows
                currBCols = propBCols
            end            
        end
        
        #Inner iteration for M
        #println("Moving M")
        for mm in nM
            propM, ratioM = transitionM(currM)

            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioM

            #accept / reject move
            if rand() < a1 * a2
                currM = propM
            end
        end

        #Inner iteration for U
        #println("Moving U")
        for uu in nU
            propU, ratioU = transitionU(currU)

            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioU

            #accept / reject move
            if rand() < a1 * a2
                currU = propU
            end
        end        
    end

    return currBM, currPerm, currM, currU
end
