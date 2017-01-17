#table
#  U  M
#0
#1

function datatotable{G <: Integer}(data::Array{G, 3}, C::MatchMatrix)
    if size(data, 2) != C.nrow
        error("second dimention of data must match number of rows in MatchMatrix")
    end
    if size(data, 3) != C.ncol
        error("third dimention of data must match number of columns in MatchMatrix")
    end
    nmeasure = size(data, 1)
    nobs = length(data[1, :, :])
    nones = length(C.rows)
    datatable = zeros(G, nmeasure, 2, 2)
    for ii in 1:nmeasure
            for kk in 1:nones
                if data[ii, C.rows[kk], C.cols[kk]] == 1
                    datatable[ii, 2, 2] += 1
                end
            end
        datatable[ii, 1, 2] = nones - datatable[ii, 2, 2]
        datatable[ii, 2, 1] = sum(data[ii, :, :]) - datatable[ii, 2, 2]
        datatable[ii, 1, 1] = nobs - sum(datatable[ii, :, :])
        
    end
    return datatable
end

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

function metropolis_hastings{G <: Integer, T <: AbstractFloat}(niter::Int64, data::Array{G, 3}, C0::MatchMatrix, M0::Array{T, 1}, U0::Array{T, 1}, logpdfC::Function, logpdfM::Function, logpdfU::Function, loglikelihood::Function, transitionC::Function, transitionMU::Function,transitionC_ratio::Function, transitionMU_ratio::Function)
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

function countones{G <: Integer}(data::Array{G, 3})
    out = Array{Int64}(size(data)[2:3]...)
    for ii in eachindex(out)
        out[ii] = sum(data[:, ii])
    end
    return out
end

function countones{G <: Integer, T <: AbstractFloat}(data::Array{G, 3}, weights::Array{T, 1})
    out = Array{Int64}(size(data)[2:3]...)
    for ii in eachindex(out)
        out[ii] = dot(data[:, ii], weights)
    end
    return out
end

function totalmatches(x::Array{MatchMatrix, 1})
    n = length(x)
    nmatches = zeros(Int64, n)
    for ii in 1:n
        nmatches[ii] = length(x[ii].rows)
    end
    totals = zeros(Int64, x[1].nrow, x[1].ncol)
    for c in x
        for ii in 1:length(c.rows)
            totals[c.rows[ii], c.cols[ii]] += 1
        end
    end
    return nmatches, totals ./ n
end

function metropolis_hastings_mixing{G <: Integer, T <: AbstractFloat}(niter::Int64,
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
                                                                      transitionMU_ratio::Function;
                                                                      nC::Int64 = 1,
                                                                      nM::Int64 = 1,
                                                                      nU::Int64 = 1)
    CArray = Array{MatchMatrix}(niter)
    MArray = Array{eltype(M0)}(niter, length(M0))
    UArray = Array{eltype(U0)}(niter, length(U0))

    ii = 1
    currC = C0
    currTable = datatotable(curr, C0)
    currM = M0
    currU = U0
        
    logP = logpdfC(C0) + logpdfM(M0) + logpdfU(U0) + loglikelihood(datatable0, M0, U0)

    #out iteration
    for ii in 1:niter
        
        #Inner iteration for C
        for cc in 1:nC
            propC = transitionC(currC)
            propTable = datatotable(data, propC)
            #compute a1
            a1 = exp(logpdfC(propC) + loglikelihood(propTable, currM, currU) - logpdfC(currC) - loglikelihood(currTable, currM, currU))
            #compute a2
            a2 = transitionC_ratio(currC, propC)
            if rand() < a1 * a2
                currC = probC
                currTable = propTable
            end
        end

        #Inner iteration for M
        for mm in nM
            propM = transitionMU(currM)
            #compute a1
            a1 = exp(logpdfM(propM) + loglikelihood(currTable, propM, currU) - logpdfM(currM) - loglikelihood(currTable, currM, currU))
            #compute a2
            a2 = transitionMU_ratio(MArray[ii, :], propM)
            if rand() < a1 * a2
                currM = propM
            end
        end

        #Inner iteration for M
        for uu in nU
            propU = transitionMU(currU)
            #compute a1
            a1 = exp(logpdfU(propU) + loglikelihood(currTable, currM, propU) - logpdfU(currU) - loglikelihood(currTable, currM, currU))
            #compute a2
            a2 = transitionMU_ratio(UArray[ii, :], propU)
            if rand() < a1 * a2
                currU = propU
            end
        end

        #Add states to chain
        CArray[ii] = currC
        MArray[ii, :] = currM
        UArray[ii, :] = currU
    end
    return CArray, MArray, UArray
end
