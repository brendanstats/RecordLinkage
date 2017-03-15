"""
Type to define the matching matrix in record linkage problem
"""
type MatchMatrix{G <: Integer}
    rows::Array{G, 1}
    cols::Array{G, 1}
    nrow::G
    ncol::G
end

#Need to check
MatchMatrix{G <: Integer}(nrow::G, ncol::G) = MatchMatrix(Array{G}(0), Array{G}(0), nrow, ncol)

MatchMatrix{G <: Intger}(GM::GridMatchMatrix{G}) = MatchMatrix(getmatches(GM)..., GM.nrow, GM.ncol)

"""
Check if supplied row and column correspond to a match in the matching matrix
"""
function Base.in(M::MatchMatrix, row::Int64, col::Int64)
    if row > M.nrow || col > M.ncol
        return error("Row or column outside matrix bounds")
    end
    if  row <= 0 || col <= 0
        return error("Row and column must be positive")
    end

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
        if row == M.rows[ii] && col == M.cols[ii]
            return ii
        end
    end
    return 0
end

"""
Add a match to MatchMatrix
"""
function add_match!(M::MatchMatrix, row::Int64, col::Int64; check::Bool = false)
    #check if it already exits
    if check
        if in(M, row, col)
            println("Entry already exists, no change made")
            return M
        end
    end
    
    #check if outside bounds
    if row > M.nrow || col > M.ncol
        error("Attempt to index outside of allowable range")
    end
    push!(M.rows, row)
    push!(M.cols, col)
    return M
end

"""
Remove a match from a MatchMatrix
"""
function remove_match!(M::MatchMatrix, row::Int64, col::Int64)
    idx = findindex(M, row, col)
    #check if match exists
    if idx == 0
        println("Does not  exists, no change made")
        return M
    end
    
    #find index
    deleteat!(M.rows, idx)
    deleteat!(M.cols, idx)
    return M
end

function convert{G <: Integer}(::Type{Array{G, 2}}, M::MatchMatrix)
    out = zeros(G, M.nrow, M.ncol)
    for ii in 1:length(M.rows)
        out[M.rows[ii], M.cols[ii]] = 1
    end
    return out
end

"""
Return rows with no matches
"""
function empty_rows(M::MatchMatrix)
    return setdiff(1:M.nrow, M.rows)
end

"""
Return columns with no matches
"""
function empty_cols(M::MatchMatrix)
    return setdiff(1:M.ncol, M.cols)
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

function move_matchmatrix_col!(M::MatchMatrix, p::AbstractFloat)
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

"""
Move a MatchMatrix changing the structure
"""
function move_matchmatrix!(M::MatchMatrix, p::AbstractFloat)
    if p < 0.0 || p > 1.0
        error("p must be between 0 and 1")
    end
    if M.nrow <= M.ncol
        return move_matchmatrix_row!(M, p)
    else
        return move_matchmatrix_col!(M, p)
    end
end

function copy(M::MatchMatrix)
    return MatchMatrix(Base.copy(M.rows), Base.copy(M.cols), Base.copy(M.nrow), Base.copy(M.ncol))
end

"""
Performs a move on a copy of a MatchMatrix changing the structure
"""
function move_matchmatrix(M::MatchMatrix, p::AbstractFloat)
    #return move_matchmatrix!(MatchMatrix(copy(M.rows), copy(M.cols), copy(M.nrow), copy(M.ncol)), p)
    return move_matchmatrix!(copy(M), p)
end

"""
Find the ratio
``\frac{g(M1|M2)}{g(M2|M1)}``
"""
function ratio_pmove_row(M1::MatchMatrix, M2::MatchMatrix, p::AbstractFloat)
    matchchange = length(M2.rows) - length(M1.rows)
    if matchchange == 1
        return p * (M1.ncol - length(M1.rows))
    elseif matchchange == -1
        return  1.0 / (p * (M1.ncol - length(M2.rows)))
    elseif matchchange == 0
        return 1.0
    else
        return 0.0
    end
end

"""
Find the ratio
``\frac{g(M1|M2)}{g(M2|M1)}``
"""
function ratio_pmove_col(M1::MatchMatrix, M2::MatchMatrix, p::AbstractFloat)
    matchchange = length(M2.rows) - length(M1.rows)
    if matchchange == 1
        return p * (M1.nrow - length(M1.cols))
    elseif matchchange == -1
        return  1.0 / (p * (M1.nrow - length(M2.cols)))
    elseif matchchange == 0
        return 1.0
    else
        return 0.0
    end
end

"""
Find the ratio
``\frac{g(M1|M2)}{g(M2|M1)}``
"""
function ratio_pmove(M1::MatchMatrix, M2::MatchMatrix, p::AbstractFloat)
    if p < 0.0 || p > 1.0
        error("p must be between 0 and 1")
    end
    if M1.nrow != M2.nrow || M1.ncol != M2.ncol
        error("MatchMatrixes must have the same dimension")
    end
    if M1.nrow <= M1.ncol
        return ratio_pmove_row(M1, M2, p)
    else
        return ratio_pmove_col(M1, M2, p)
    end
end

"""
Return a nx2 array with the first column containing row indicies and second containing column indicies
"""
function match_pairs(M::MatchMatrix)
    return [M.rows M.cols]
end

"""
Compute the total number of matches that occured in each entry of an Array of MatchMatrixies
"""
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

function =={M <: MatchMatrix}(M1::M, M2::M)
    if M1.nrow != M2.nrow
        return false
    elseif M1.ncol != M2.ncol
        return false
    elseif length(M1.rows) != length(M2.rows)
        return false
    else
        perm1 = sortperm(M1.rows)
        perm2 = sortperm(M2.rows)
        if M1.rows[perm1] != M2.rows[perm2]
            return false
        elseif M1.cols[perm1] != M2.cols[perm2]
            return false
        else
            return true
        end
    end
end
