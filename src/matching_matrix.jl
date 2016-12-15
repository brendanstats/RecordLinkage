"""
Type to define the matching matrix in record linkage problem
"""
type MatchMatrix
    rows::Array{Int64, 1}
    cols::Array{Int64, 1}
    nrow::Int64
    ncol::Int64
end

"""
Check if supplied row and column correspond to a match in the matching matrix
"""
function in(M::MatchMatrix, row::Int64, col::Int64)
    if row > M.nrow || col > M.ncol
        error("Row or column outside matrix bounds")
    for ii in 1:length(M.rows)
        if row == M.rows[ii] && col == M.cols[ii]
            return true
        end
    end
    return false
end

"""
Find the index corresponding to the row and col in the matching matrix
"""
function findindex(M::MatchMatrix, row::Int64, col::Int64)
    for ii in 1:length(M.rows)
        if x == M.rows[ii] && y == M.cols[ii]
            return ii
        end
    end
    return 0
end

#findnext

function add_match!(M::MatchMatrix, row::Int64, col::Int64; check::Bool = false)
    #check if it already exits
    if check
        if in(M, row, col)
            println("Entry already exists, no change made")
            return M
        end
    end
    
    #check if outside bounds
    if x > M.nrow || y > M.ncol
        error("Attempt to index outside of allowable range")
    end
    push!(M.rows, x)
    push!(M.cols, y)
    return M
end

function remove_match!(M::MatchMatrix, row::Int64, col::Int64)
    idx = findindex(M, row, col)
    #check if match exists
    #find index
    deleteat!()
    deleteat!()
    return M
end

function convert{G <: Integer}(T::Array{G, 2}, M::MatchMatrix)
    out = zeros(G, M.nrow, M.ncol)
    for ii in 1:length(M.rows)
        out[M.rows[ii], M.cols[ii]] = 1
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
function move_matchmatrix_row!(M::MatchMatrix, p::AbstractFloat)
    r = StatsBase.sample(1:M.nrow)
    idx = findnext(M.rows, r, 1)
    if idx == 0
        c = StatsBase.sample(empty_cols(M))
        push!(M.rows, r)
        push!(M.cols, c)
    else
        if rand() < p
            deleteat!(M.rows, idx)
            deleteat!(M.cols, idx)
        else
            M.rows[idx] = StatsBase.sample([M.rows[idx]; empty_rows(M)])
            M.cols[idx] = StatsBase.sample([M.cols[idx]; empty_cols(M)])
        end
    end
    return M
end

function move_matchmatrix_col(M::MatchMatrix, p::AbstractFloat)
    c = StatsBase.sample(1:M.ncol)
    idx = findnext(M.cols, c, 1)
    if idx == 0
        r = StatsBase.sample(empty_rows(M))
        push!(M.rows, r)
        push!(M.cols, c)
    else
        if rand() < p
            deleteat!(M.rows, idx)
            deleteat!(M.cols, idx)
        else
            M.rows[idx] = StatsBase.sample([M.rows[idx]; empty_rows(M)])
            M.cols[idx] = StatsBase.sample([M.cols[idx]; empty_cols(M)])
        end
    end
    return M
end


function move_matchmatrix(M::MatchMatrix, p::AbstractFloat)
    if p < 0.0 || p > 1.0
        error("p must be between 0 and 1")
    end
    if M.nrow <= M.ncol
        move_matchmatrix_row(M, p)
    else
        move_matchmatrix_col(M, p)
    end
end
