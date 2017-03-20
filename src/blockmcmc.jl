"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a grid of MatchMatricies
"""
function metropolis_hastings_mixing{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::BitArray{3},
    blockrows::Array{G, 1},
    blockcols::Array{G, 1},
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
    
    #MCMC Chains
    #BMArray = Array{BlockMatchMatrix}(niter, length(blockrows))
    BMArray = Array{MatchMatrix{G}}(niter, length(blockrows))
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    #Track transitions
    transBM = falses(niter)
    transM = falses(niter)
    transU = falses(niter)

    #Initial States
    currBM = BM0
    currTable = data2table(data, blockrows, blockcols, currBM)
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for BM
        for gg in 1:nBM
            propBM, ratioBM = transitionBM(blockrows, blockcols, currBM)
            propTable = data2table(data, blockrows, blockcols, propBM)
            #println("prop C")
            #compute a1
            a1 = exp(logpdfBM(blockrows, blockcols, propBM) + loglikelihood(propTable, currM, currU) - logpdfBM(blockrows, blockcols, currBM) - loglikelihood(currTable, currM, currU))
            #println("a1 C")
            #compute a2
            a2 = ratioBM
            #println("a2 C")
            if rand() < a1 * a2
                #println("BlockMatrixTransition")
                currBM = propBM
                currTable = propTable
                transBM[ii] = true
            end
        end

        #Inner iteration for M
        for mm in nM
            propM, ratioM = transitionM(currM)
            #println("prop M")
            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))
            #println("a1 M")
            #compute a2
            a2 = ratioM
            #println("a2 M")
            if rand() < a1 * a2
                currM = propM
                transM[ii] = true
            end
        end

        #Inner iteration for M
        for uu in nU
            propU, ratioU = transitionU(currU)
            #println("prop U")
            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))
            #println("a1 U")
            #compute a2
            a2 = ratioU
            #println("a2 U")
            if rand() < a1 * a2
                currU = propU
                transU[ii] = true
            end
        end

        #Add states to chain
        for (jj, (rr, cc)) in enumerate(zip(blockrows, blockcols))
            BMArray[ii, jj] = currBM.blocks[rr, cc]
        end
        MArray[ii, :] = currM
        UArray[ii, :] = currU
    end
    return BMArray, MArray, UArray, transBM, transM, transU
end

"""
Allow rows to be excluded
"""
function metropolis_hastings_mixing{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::BitArray{3},
    blockrows::Array{G, 1},
    blockcols::Array{G, 1},
    exrows::Array{G, 1},
    excols::Array{G, 1},
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
    
    #MCMC Chains
    #BMArray = Array{BlockMatchMatrix}(niter, length(blockrows))
    BMArray = Array{MatchMatrix{G}}(niter, length(blockrows))
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    #Track transitions
    transBM = falses(niter)
    transM = falses(niter)
    transU = falses(niter)

    #Initial States
    currBM = BM0
    currTable = data2table(data, blockrows, blockcols, currBM)
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for BM
        for gg in 1:nBM
            propBM, ratioBM = transitionBM(blockrows, blockcols, exrows, excols, currBM)
            propTable = data2table(data, blockrows, blockcols, propBM)
            #println("prop C")
            #compute a1
            a1 = exp(logpdfBM(blockrows, blockcols, propBM) + loglikelihood(propTable, currM, currU) - logpdfBM(blockrows, blockcols, currBM) - loglikelihood(currTable, currM, currU))
            #println("a1 C")
            #compute a2
            a2 = ratioBM
            #println("a2 C")
            if rand() < a1 * a2
                currBM = propBM
                currTable = propTable
                transBM[ii] = true
            end
        end

        #Inner iteration for M
        for mm in nM
            propM, ratioM = transitionM(currM)
            #println("prop M")
            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))
            #println("a1 M")
            #compute a2
            a2 = ratioM
            #println("a2 M")
            if rand() < a1 * a2
                currM = propM
                transM[ii] = true
            end
        end

        #Inner iteration for M
        for uu in nU
            propU, ratioU = transitionU(currU)
            #println("prop U")
            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))
            #println("a1 U")
            #compute a2
            a2 = ratioU
            #println("a2 U")
            if rand() < a1 * a2
                currU = propU
                transU[ii] = true
            end
        end

        #Add states to chain
        for (jj, (rr, cc)) in enumerate(zip(blockrows, blockcols))
            BMArray[ii, jj] = currBM.blocks[rr, cc]
        end
        MArray[ii, :] = currM
        UArray[ii, :] = currU
    end
    return BMArray, MArray, UArray, transBM, transM, transU
end

"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a grid of MatchMatricies
"""
function metropolis_hastings_sample{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::BitArray{3},
    blockrows::Array{G, 1},
    blockcols::Array{G, 1},
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
    
    #Initial States
    currBM = BM0
    currTable = data2table(data, blockrows, blockcols, currBM)
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for BM
        for gg in 1:nBM
            propBM, ratioBM = transitionBM(blockrows, blockcols, currBM)
            propTable = data2table(data, blockrows, blockcols, propBM)

            #compute a1
            a1 = exp(logpdfBM(blockrows, blockcols, propBM) + loglikelihood(propTable, currM, currU) - logpdfBM(blockrows, blockcols, currBM) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioBM

            #check if move accepted
            if rand() < a1 * a2
                currBM = propBM
                currTable = propTable
            end
        end

        #Inner iteration for M
        for mm in nM
            propM, ratioM = transitionM(currM)

            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioM

            #check if move accepted
            if rand() < a1 * a2
                currM = propM
            end
        end

        #Inner iteration for M
        for uu in nU
            propU, ratioU = transitionU(currU)

            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioU

            #check if move accepted
            if rand() < a1 * a2
                currU = propU
                transU[ii] = true
            end
        end
    end
    return currBM, currM, currU
end

"""
Allow rows to be excluded
"""
function metropolis_hastings_sample{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::BitArray{3},
    blockrows::Array{G, 1},
    blockcols::Array{G, 1},
    exrows::Array{G, 1},
    excols::Array{G, 1},
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
    
    #Initial States
    currBM = BM0
    currTable = data2table(data, blockrows, blockcols, currBM)
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for BM
        for gg in 1:nBM
            propBM, ratioBM = transitionBM(blockrows, blockcols, exrows, excols, currBM)
            propTable = data2table(data, blockrows, blockcols, propBM)

            #compute a1
            a1 = exp(logpdfBM(blockrows, blockcols, propBM) + loglikelihood(propTable, currM, currU) - logpdfBM(blockrows, blockcols, currBM) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioBM

            #check for transition
            if rand() < a1 * a2
                currBM = propBM
                currTable = propTable
            end
        end

        #Inner iteration for M
        for mm in nM
            propM, ratioM = transitionM(currM)

            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioM

            #check for transition
            if rand() < a1 * a2
                currM = propM
            end
        end

        #Inner iteration for M
        for uu in nU
            propU, ratioU = transitionU(currU)

            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))

            #compute a2
            a2 = ratioU

            #check for transition
            if rand() < a1 * a2
                currU = propU
            end
        end
    end
    return currBM, currM, currU
end
