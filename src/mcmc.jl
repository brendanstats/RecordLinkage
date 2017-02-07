
function loglikelihood_datatable{G <: Integer, T <: AbstractFloat}(datatable::Array{G, 3}, γM::Array{T, 1}, γU::Array{T, 1})
    if any(γM .< 0.0) || any(γM .> 1.0)
        error("M probabilities must be between 0 and 1")
    end
    if any(γU .< 0.0) || any(γU .> 1.0)
        error("U probabilities must be between 0 and 1")
    end
    n = length(γM)
    if n != length(γU)
        error("Length of M and U probabilities do not match")
    end
    if size(datatable) != (n, 2, 2)
        error("Input data must have first dimention match length of probabilities and other two dimentions be length 2")
    end
    loglike = 0.0
    for ii in 1:n
        loglike += log(1.0 - γU[ii]) * datatable[ii, 1, 1]
        loglike += log(1.0 - γM[ii]) * datatable[ii, 1, 2]
        loglike += log(γU[ii]) * datatable[ii, 2, 1]
        loglike += log(γM[ii]) * datatable[ii, 2, 2]
    end
    return loglike
end

"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a MatchMatrix
"""
function metropolis_hastings{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::Array{G, 3},
    C0::MatchMatrix,
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfC::Function,
    logpdfM::Function,
    logpdfU::Function,
    loglikelihood::Function,
    transitionC::Function,
    transitionMU::Function,
    transitionC_ratio::Function,
    transitionMU_ratio::Function)
    
    CArray = Array{MatchMatrix}(niter)
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    ii = 1
    CArray[ii] = C0
    MArray[ii, :] = M0
    UArray[ii, :] = U0
    datatable0 = datatotable(data, C0)
    
    logP = logpdfC(C0) + logpdfM(M0) + logpdfU(U0) + loglikelihood(datatable0, M0, U0)

    while ii < niter
        #draw proposal
        propC = transitionC(CArray[ii])
        propDatatable = datatotable(data, propC)
        propM = transitionMU(MArray[ii, :])
        propU = transitionMU(UArray[ii, :])

        #compute a1
        proplogP = logpdfC(propC) + logpdfM(propM) + logpdfU(propU) + loglikelihood(propDatatable, propM, propU)
        a1 = exp(proplogP - logP)
        
        #compute a2
        a2 = transitionC_ratio(CArray[ii], propC) * transitionMU_ratio(MArray[ii, :], propM) * transitionMU_ratio(UArray[ii, :], propU)
        ii += 1
        if rand() < a1 * a2            
            #update parameters
            CArray[ii] = propC
            MArray[ii, :] = propM
            UArray[ii, :] = propU
            
            #update probabilities
            logP = proplogP
        else
            CArray[ii] = CArray[ii - 1]
            MArray[ii, :] = MArray[ii - 1, :]
            UArray[ii, :] = UArray[ii - 1, :]
        end
    end
    return CArray, MArray, UArray
end

"""
Metropolis-Hastings MCMC Algorithm for posterior distribution of a MatchMatrix with independent mixing
"""
function metropolis_hastings_mixing{G <: Integer, T <: AbstractFloat}(
    niter::Int64,
    data::Array{G, 3},
    C0::MatchMatrix,
    M0::Array{T, 1},
    U0::Array{T, 1},
    logpdfC::Function,
    logpdfM::Function,
    logpdfU::Function,
    loglikelihood::Function,
    transitionC::Function,
    transitionM::Function,
    transitionU::Function,
    transitionC_ratio::Function,
    transitionM_ratio::Function,
    transitionU_ratio::Function;
    nC::Int64 = 1,
    nM::Int64 = 1,
    nU::Int64 = 1)
    
    #MCMC Chains
    CArray = Array{MatchMatrix}(niter)
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    #Track transitions
    transC = falses(niter)
    transM = falses(niter)
    transU = falses(niter)

    #Initial States
    currC = C0
    currTable = datatotable(data, currC)
    currM = M0
    currU = U0
    
    #out iteration
    for ii in 1:niter
        
        #Inner iteration for C
        for cc in 1:nC
            propC = transitionC(currC)
            propTable = datatotable(data, propC)
            #println("prop C")
            #compute a1
            a1 = exp(logpdfC(propC) + loglikelihood(propTable, currM, currU) - logpdfC(currC) - loglikelihood(currTable, currM, currU))
            #println("a1 C")
            #compute a2
            a2 = transitionC_ratio(currC, propC)
            #println("a2 C")
            if rand() < a1 * a2
                currC = propC
                currTable = propTable
                transC[ii] = true
            end
        end

        #Inner iteration for M
        for mm in nM
            propM = transitionM(currM)
            #println("prop M")
            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))
            #println("a1 M")
            #compute a2
            a2 = transitionM_ratio(currM, propM)
            #println("a2 M")
            if rand() < a1 * a2
                currM = propM
                transM[ii] = true
            end
        end

        #Inner iteration for M
        for uu in nU
            propU = transitionU(currU)
            #println("prop U")
            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))
            #println("a1 U")
            #compute a2
            a2 = transitionU_ratio(currU, propU)
            #println("a2 U")
            if rand() < a1 * a2
                currU = propU
                transU[ii] = true
            end
        end

        #Add states to chain
        CArray[ii] = currC
        MArray[ii, :] = currM
        UArray[ii, :] = currU
    end
    return CArray, MArray, UArray, transC, transM, transU
end
