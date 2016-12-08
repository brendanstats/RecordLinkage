"""
Type to define the matching matrix in record linkage problem
"""
type MatchMatrix
    row::Array{Int64, 1}
    col::Array{Int64, 1}
    nrows::Int64
    ncols::Int64
end

function in(M::MatchMatrix, x::Int64, y::Int64)
    for ii in 1:length(M.row)
        if x == M.row[ii] && y == M.col[ii]
            return true
        end
    end
    return false
end

function findindex(M::MatchMatrix, x::Int64, y::Int64)
    for ii in 1:length(M.row)
        if x == M.row[ii] && y == M.col[ii]
            return true
        end
    end
    return false
end


function add_match!(M::MatchMatrix, x::Int64, y::Int64; check = false)
    #check if it already exits
    if check
        if in(M, x, y)
            println("Entry all ready exists, no change made")
            return M
        end
    end
    
    #check if outside bounds
    if x > M.nrow || y > M.ncol
        error("Attempt to index outside of allowable range")
    end
    push!(M.row, x)
    push!(M.col, y)
    return M
end

function remove_match!(M::MatchMatrix, x::Int64, y::Int64)
    #check if match exists
    slice!()
    slice!()
    return M
end

function remove_match!(M::MatchMatrix, x::Int64)
    #check if match exists
    slice!()
    slice!()
    return M
end


function convert(Array{Int64, 2}, M::MatchMatrix)
    out = zeros(Int64, M.nrow, M.ncol)
    for ii in 1:length(M.row)
        out[M.row[ii], M.col[ii]] = 1
    end
    return out
end

function empty_rows(M::MatchMatrix)
    return setdiff(1:M.nrow, M.row)
end

function empty_cols(M::MatchMatrix)
    return setdiff(1:M.ncol, M.col)
end

#Liseo and Tancredi: Bayesian Estimation of Population Size
#select row randomly
#if no entry in row 
#delete entry if it has one (p) or move (1-p)
function move_matchmatrix(M::MatchMatrix, p::Float64)
    if p < 0.0 || p > 1.0
        error("p must be between 0 and 1")
    end
    if M.nrow <= M.ncol
        #choose a row
        ii = StatsBase.sample(1:M.nrow)
        #if it has an entry eithrt delete or move it
        if in(ii, M.row)

        else
            #create a new entry
        end
    else
    end
end
            
