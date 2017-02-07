#table
#  U  M
#0
#1

"""
Compute 2x2 table of data for each measure
"""
function datatotable{G <: Integer}(data::Array{G, 3}, C::MatchMatrix)
    if size(data, 2) != C.nrow
        error("second dimention of data must match number of rows in MatchMatrix")
    end
    if size(data, 3) != C.ncol
        error("third dimention of data must match number of columns in MatchMatrix")
    end
    nmeasure = size(data, 1)
    nobs = length(data[1, :, :])

    #number of true mathces
    nones = length(C.rows)
    datatable = zeros(Int64, nmeasure, 2, 2)

    for ii in 1:nmeasure
        #number of observed matches that are true matches 
        for kk in 1:nones
            if data[ii, C.rows[kk], C.cols[kk]] == 1
                datatable[ii, 2, 2] += 1
            end
        end
        #unobserved matches = true matches - observed true matches
        datatable[ii, 1, 2] = nones - datatable[ii, 2, 2]
        #false matches = observed matches - observed true matches
        datatable[ii, 2, 1] = sum(data[ii, :, :]) - datatable[ii, 2, 2]
        #true nonmatches = observations - other categories
        datatable[ii, 1, 1] = nobs - sum(datatable[ii, :, :])        
    end
    return datatable
end

"""
Compute 2x2 table of data for each measure
"""
function datatotable{G <: Integer}(data::Array{G, 3}, GM::GridMatchMatrix)
    if size(data, 2) != GM.nrow
        error("second dimention of data must match number of rows in GridMatchMatrix")
    end
    if size(data, 3) != GM.ncol
        error("third dimention of data must match number of columns in GridMatchMatrix")
    end
    nmeasure = size(data, 1)
    matchrows, matchcols = getmatches(GM)
    nobs = length(data[1, :, :])
    nones = length(matchrows)
    datatable = zeros(Int64, nmeasure, 2, 2)
    for ii in 1:nmeasure
        for (rr, cc) in zip(matchrows, matchcols)
            if data[ii, rr, cc] == 1
                datatable[ii, 2, 2] += 1
            end
        end
        #unobserved matches = true matches - observed true matches
        datatable[ii, 1, 2] = nones - datatable[ii, 2, 2]
        #false matches = observed matches - observed true matches
        datatable[ii, 2, 1] = sum(data[ii, :, :]) - datatable[ii, 2, 2]
        #true nonmatches = observations - other categories
        datatable[ii, 1, 1] = nobs - sum(datatable[ii, :, :])        
    end
    return datatable
end

function datatotable{G <: Integer, T <: Integer}(data::Array{G, 3}, grows::Array{T, 1}, gcols::Array{T, 1}, GM::GridMatchMatrix)
    if size(data, 2) != GM.nrow
        error("second dimention of data must match number of rows in GridMatchMatrix")
    end
    if size(data, 3) != GM.ncol
        error("third dimention of data must match number of columns in GridMatchMatrix")
    end
    nmeasure = size(data, 1)
    matchrows, matchcols = getmatches(grows, gcols, GM)
    nobs = length(data[1, :, :])
    nones = length(matchrows)
    datatable = zeros(Int64, nmeasure, 2, 2)
    for ii in 1:nmeasure
        for (rr, cc) in zip(matchrows, matchcols)
            if data[ii, rr, cc] == 1
                datatable[ii, 2, 2] += 1
            end
        end
        #unobserved matches = true matches - observed true matches
        datatable[ii, 1, 2] = nones - datatable[ii, 2, 2]
        #false matches = observed matches - observed true matches
        datatable[ii, 2, 1] = sum(data[ii, :, :]) - datatable[ii, 2, 2]
        #true nonmatches = observations - other categories
        datatable[ii, 1, 1] = nobs - sum(datatable[ii, :, :])        
    end
    return datatable
end

"""
Loglikelihood of observed datatable give M and U probabilities
"""
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
Count the number of ones in a 3D Array
"""
function countones{G <: Integer}(data::Array{G, 3})
    out = Array{Int64}(size(data)[2:3]...)
    for ii in eachindex(out)
        out[ii] = sum(data[:, ii])
    end
    return out
end

"""
Count the number of ones in a 3D Array with weighting
"""
function countones{G <: Integer, T <: AbstractFloat}(data::Array{G, 3}, weights::Array{T, 1})
    out = Array{Int64}(size(data)[2:3]...)
    for ii in eachindex(out)
        out[ii] = dot(data[:, ii], weights)
    end
    return out
end
