function datatotable(data::Array{Integer, 3}, C::MatchMatrix)
    if size(data, 2) != C.nrow
        error("second dimention of data must match number of rows in MatchMatrix")
    end
    if size(data, 3) != C.ncol
        error("third dimention of data must match number of columns in MatchMatrix")
    end
    N = length(data[1, :, :])
    datatable = zeros()
end
#table
#  U  M
#0
#1

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
