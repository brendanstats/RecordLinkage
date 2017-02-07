"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a grid of MatchMatricies
"""
function metropolis_hastings_mixing{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::Array{G, 3},
    grows::Array{Int64, 1},
    gcols::Array{Int64, 1},
    GM0::GridMatchMatrix,
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
    
    #MCMC Chains
    GMArray = Array{GridMatchMatrix}(niter, length(grows))
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    #Track transitions
    transGM = falses(niter)
    transM = falses(niter)
    transU = falses(niter)

    #Initial States
    currGM = GM0
    currTable = datatotable(data, grows, gcols, currGM)
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for GM
        for gg in 1:nGM
            propGM, ratioGM = transitionGM(grows, gcols, currGM)
            propTable = datatotable(data, propGM)
            #println("prop C")
            #compute a1
            a1 = exp(logpdfGM(propGM) + loglikelihood(propTable, currM, currU) - logpdfGM(currGM) - loglikelihood(currTable, currM, currU))
            #println("a1 C")
            #compute a2
            a2 = ratioGM
            #println("a2 C")
            if rand() < a1 * a2
                currGM = propGM
                currTable = propTable
                transGM[ii] = true
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
        GMArray[ii, :] = currGM
        MArray[ii, :] = currM
        UArray[ii, :] = currU
    end
    return GMArray, MArray, UArray, transGM, transM, transU
end

"""
Allow rows to be excluded
"""
function metropolis_hastings_mixing{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::Array{G, 3},
    grows::Array{Int64, 1},
    gcols::Array{Int64, 1},
    exrows::Array{Int64, 1},
    excols::Array{Int64, 1},
    GM0::GridMatchMatrix,
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
    
    #MCMC Chains
    GMArray = Array{GridMatchMatrix}(niter, length(grows))
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    #Track transitions
    transGM = falses(niter)
    transM = falses(niter)
    transU = falses(niter)

    #Initial States
    currGM = GM0
    currTable = datatotable(data, grows, gcols, currGM)
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for GM
        for gg in 1:nGM
            propGM, ratioGM = transitionGM(grows, gcols, exrows, excols, currGM)
            propTable = datatotable(data, propGM)
            #println("prop C")
            #compute a1
            a1 = exp(logpdfGM(propGM) + loglikelihood(propTable, currM, currU) - logpdfGM(currGM) - loglikelihood(currTable, currM, currU))
            #println("a1 C")
            #compute a2
            a2 = ratioGM
            #println("a2 C")
            if rand() < a1 * a2
                currGM = propGM
                currTable = propTable
                transGM[ii] = true
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
        GMArray[ii, :] = currGM
        MArray[ii, :] = currM
        UArray[ii, :] = currU
    end
    return GMArray, MArray, UArray, transGM, transM, transU
end
