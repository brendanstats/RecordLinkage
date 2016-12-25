#table
#  U  M
#0
#1

function datatotable(data::Array{Integer, 3}, C::MatchMatrix)
    if size(data, 2) != C.nrow
        error("second dimention of data must match number of rows in MatchMatrix")
    end
    if size(data, 3) != C.ncol
        error("third dimention of data must match number of columns in MatchMatrix")
    end
    nmeasure = size(data, 1)
    nobs = length(data[1, :, :])
    nones = length(C.rows)
    datatable = zeros(nmeasure, 2, 2)
    for ii in 1:nmeasure
            for kk in 1:nones
                if data[ii, C.rows[kk], C.cols[kk]] == 1
                    datatable[ii, 2, 2] += 1
                end
                datatable[ii, 1, 2] = nones - datatable[ii, 2, 2]
                datatable[ii, 2, 1] = sum(data[ii, :, :]) - datatable[ii, 2, 2]
                datatable[ii, 1, 1] = nobs - sum(datatable[ii, :, :])
        end
    end
    return datatable
end

function loglikelihood(datatable::Array{Integer, 3}, γM::Array{AbstractFloat, 1},
                       γU::Array{AbstractFloat, 1})
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

function metropolis_hastings(niter::Integer, data::Array{AbstractFloat, 3},
                              C0::MatchMatrix, M0::Array{AbstractFloat, 1}, U0::Array{AbstractFloat, 1},
                             logpdfC::Function, logpdfM::Function, logpdfU::Function, loglikelihood::Function,
                             transitionC::Function, transitionMU::Function,
                             transitionC_ratio::Function, transitionMU_ratio::Function)
    CArray = Array{MatchMatrix}(niter)
    γMArray = Array{eltype(γM0)}(niter, length(γM0))
    γUArray = Array{eltype(γU0)}(niter, length(γU0))

    ii = 1
    CArray[ii] = C0
    γMArray[ii, :] = γM0
    γUArray[ii, :] = γU0
    datatable0 = datatotable(data, C0)
    
    logP = logpdfC(C0) + logpdfM(γM0) + logpdfM(γU0) + loglikelihood(datatable0)

    while ii < niter
        #draw proposal
        propC = transitionC(CArray[ii])
        propDatatable = datatotable(data, propC)
        propγM = transitionγ(γM0)
        propγU = transitionγ(γU0)

        #compute a1
        proplogP = logpdfC(propC) + logpdfM(propγM) + logpdfM(propγU) + loglikelihood(propDatatable)
        a1 = exp(proplogP - logP)
        
        #compute a2
        a2 = transitionC_ratio(CArray[ii], propC) * transitionγ_(γMArray[1, :], propγM) * transitionγ_(γUArray[1, :], propγU)
        ii += 1
        if rand() < a1 * a2            
            #update parameters
            CArray[ii] = propC
            γMArray[ii, :] = propγM
            γUArray[ii, :] = propγU
            
            #update probabilities
            logP = proplogP
        else
            CArray[ii] = CArray[ii - 1]
            γMArray[ii, :] = γMArray[ii - 1, :]
            γUArray[ii, :] = γUArray[ii - 1, :]
        end
    end
    return CArray, γMArray, γUArray
end