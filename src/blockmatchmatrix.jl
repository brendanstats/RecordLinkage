"""
Type to hold an Array with MatchMatrix elements recording other information, intended to be used for blocking
"""
type BlockMatchMatrix{G}
    blocks::Array{MatchMatrix{G}, 2}
    nrows::Array{G, 1}
    ncols::Array{G, 1}
    cumrows::Array{G, 1}
    cumcols::Array{G, 1}
    nrow::G
    ncol::G
end

"""
Constructor based only on block size
"""
function BlockMatchMatrix{G <: Integer}(nrows::Array{G, 1}, ncols::Array{G, 1})
    blocks = Array{MatchMatrix}(length(nrows), length(ncols))
    for jj in eachindex(ncols)
        for ii in eachindex(nrows)
            blocks[ii, jj] = MatchMatrix(nrows[ii], ncols[jj])
        end
    end
    return BlockMatchMatrix(blocks, nrows, ncols, cumsum(nrows), cumsum(ncols), sum(nrows), sum(ncols))
end

"""
Constructor based only on block size
"""
function BlockMatchMatrix{G <: Integer}(nrows::Array{G, 1}, ncols::Array{G, 1}, M::MatchMatrix)
    BM = BlockMatchMatrix(nrows, ncols)
    for ii in 1:length(M.cols)
        blockrow = getblockrow(M.rows[ii], BM)
        blockcol = getblockcol(M.cols[ii], BM)
        push!(BM.blocks[blockrow, blockcol].rows, M.rows[ii] - get(BM.cumrows, blockrow - 1, 0))
        push!(BM.blocks[blockrow, blockcol].cols, M.cols[ii] - get(BM.cumcols, blockcol - 1, 0))
    end
    return BM
end

"""
Constructor based on Array with type of MatchMatrix, row elements  must all be same height and column elements must all be same width
"""
function BlockMatchMatrix{G <: Integer}(blocks::Array{MatchMatrix{G}, 2})
    n, m = size(blocks)
    nrows = Array{G}(n)
    ncols = Array{G}(m)
    #get row dimensions
    for ii in 1:n
        nrows[ii] = blocks[ii, 1].nrow
    end
    #get col dimensions
    for jj in 1:m
        ncols[jj] = blocks[1, jj].ncol
    end
    for ii in 1:n
        for jj in 1:m
            if nrows[ii] != blocks[ii, jj].nrow
                error("dimensions of match matricies do not match within row")
            end
            if ncols[jj] != blocks[ii, jj].ncol
                error("dimensions of match matricies do not match within col")
            end
        end
    end
    return BlockMatchMatrix(blocks, nrows, ncols, cumsum(nrows), cumsum(ncols), sum(nrows), sum(ncols))
end

MatchMatrix{G <: Integer}(BM::BlockMatchMatrix{G}) = MatchMatrix(getmatches(BM)..., BM.nrow, BM.ncol)

"""
Create a shallow copy of BlockMatchMatrix
"""
function Base.copy(BM::BlockMatchMatrix)
    newblocks = Array{MatchMatrix}(length(BM.nrows), length(BM.ncols))
    for ii in eachindex(newblocks)
        newblocks[ii] = copy(BM.blocks[ii])
    end
    return BlockMatchMatrix(newblocks, copy(BM.nrows), copy(BM.ncols), copy(BM.cumrows), copy(BM.cumcols), copy(BM.nrow), copy(BM.ncol))
end

"""
Return the row of the grid of MatchMatrix corresponding to supplied row index of full matrix

`getblockrow(row, BM)`
"""
function getblockrow{G <: Integer}(row::G, BM::BlockMatchMatrix)
    return searchsortedfirst(BM.cumrows, row)
end

"""
Return the column of the grid of MatchMatrix corresponding to supplied column index of full matrix
"""
function getblockcol{G <: Integer}(col::G, BM::BlockMatchMatrix)
    return searchsortedfirst(BM.cumcols, col)
end

"""
Return the rows of the full matrix corresponding to the given indicies
"""
function getrows{G <: Integer}(blockrows::Array{G,1}, BM::BlockMatchMatrix)
    rows = Array{G}(0)
    for ii in unique(blockrows)
        append!(rows, (get(BM.cumrows, ii - 1, 0) + 1):BM.cumrows[ii])
    end
    return rows
end

function getrows{G <: Integer}(blockrow::G, BM::BlockMatchMatrix)
    return collect((get(BM.cumrows, blockrow - 1, 0) + 1):BM.cumrows[blockrow])
end

"""
Return the columns of the full matrix corresponding to the given indicies
"""
function getcols{G <: Integer}(blockcols::Array{G,1}, BM::BlockMatchMatrix)
    cols = Array{G}(0)
    for ii in unique(blockcols)
        append!(cols, (get(BM.cumcols, ii - 1, 0) + 1):BM.cumcols[ii])
    end
    return cols
end

function getcols{G <: Integer}(blockcol::G, BM::BlockMatchMatrix)
    return collect((get(BM.cumcols, blockcol - 1, 0) + 1):BM.cumcols[blockcol])
end

"""
Return the rows of the full matrix corresponding to the given indicies, excluding listed entries
"""
function getrows{G <: Integer}(blockrows::Array{G, 1}, exrows::Array{G, 1}, BM::BlockMatchMatrix)
    rows = Array{G}(0)
    for ii in unique(blockrows)
        append!(rows, (get(BM.cumrows, ii - 1, 0) + 1):BM.cumrows[ii])
    end
    return setdiff(rows, exrows)
end

function getrows{G <: Integer}(blockrow::G, exrows::Array{G, 1}, BM::BlockMatchMatrix)
    return setdiff(collect((get(BM.cumrows, blockrow - 1, 0) + 1):BM.cumrows[blockrow]), exrows)
end

"""
Return the columns of the full matrix corresponding to the given indicies, excluding listed entries
"""
function getcols{G <: Integer}(blockcols::Array{G,1}, excols::Array{G, 1}, BM::BlockMatchMatrix)
    cols = Array{G}(0)
    for ii in unique(blockcols)
        append!(cols, (get(BM.cumcols, ii - 1, 0) + 1):BM.cumcols[ii])
    end
    return setdiff(cols, excols)
end

function getcols{G <: Integer}(blockcol::G, excols::Array{G, 1}, BM::BlockMatchMatrix)
    return setdiff(collect((get(BM.cumcols, blockcol - 1, 0) + 1):BM.cumcols[blockcol]), excols)
end


"""
Return the rows and columns of the full matrix corresponding to the given indicies of grid of blocks
"""
function getindicies{G <: Integer}(blockrows::Array{G,1}, blockcols::Array{G,1}, BM::BlockMatchMatrix)
    rows = Array{G}(0)
    cols = Array{G}(0)
    for (ii, jj) in zip(blockrows, blockcols)
        append!(rows, (get(BM.cumrows, ii - 1, 0) + 1):BM.cumrows[ii])
        append!(cols, (get(BM.cumcols, jj - 1, 0) + 1):BM.cumcols[jj])
    end
    return zip(rows, cols)
end

"""
Get columns and rows containing a match
"""
function getmatches{G <: Integer}(blockrows::Array{G,1}, blockcols::Array{G,1}, BM::BlockMatchMatrix)
    rows = Array{G}(0)
    cols = Array{G}(0)
    for (ii, jj) in zip(blockrows, blockcols)
        if length(BM.blocks[ii, jj].rows) > 0
            append!(rows, BM.blocks[ii, jj].rows .+ get(BM.cumrows, ii - 1, 0))
            append!(cols, BM.blocks[ii, jj].cols .+ get(BM.cumcols, jj - 1, 0))
        end
    end
    return rows, cols
end

function getmatches(BM::BlockMatchMatrix)
    rows = Array{eltype(BM.blocks[1,1].rows)}(0)
    cols = Array{eltype(BM.blocks[1,1].cols)}(0)
    for ii in 1:size(BM.blocks, 1)
        for jj in 1:size(BM.blocks, 2)
            if length(BM.blocks[ii, jj].rows) > 0
                append!(rows, BM.blocks[ii, jj].rows .+ get(BM.cumrows, ii - 1, 0))
                append!(cols, BM.blocks[ii, jj].cols .+ get(BM.cumcols, jj - 1, 0))
            end
        end
    end
    return rows, cols
end

"""
Add a match to a BlockMatchMatrix
"""
function add_match!{G <: Integer}(BM::BlockMatchMatrix{G}, blockrow::G, blockcol::G, row::G, col::G)
    rowadj = row - get(BM.nrows, blockrow - 1, 0)
    coladj = col - get(BM.ncols, blockcol - 1, 0)
    push!(BM.blocks[blockrow, blockcol].rows, rowadj)
    push!(BM.blocks[blockrow, blockcol].cols, coladj)
    return BM
end

function add_match!{G <: Integer}(BM::BlockMatchMatrix{G}, blockrows::Array{G, 1}, blockcols::Array{G, 1}, rows::Array{G, 1}, cols::Array{G, 1})
    for (gr, gc, row, col) in zip(blockrows, blockcols, rows, cols)
        add_match!(BM, gr, gc, row, col)
    end
    return BM
end

function add_match!{G <: Integer}(BM::BlockMatchMatrix{G}, row::G, col::G)
    blockrow = getblockrow(row, BM)
    blockcol = getblockcol(col, BM)
    return add_match!(BM, blockrow, blockcol, row, col)
end

function add_match!{G <: Integer}(BM::BlockMatchMatrix{G}, rows::Array{G, 1}, cols::Array{G, 1})
    for (row, col) in zip(rows, cols)
        add_match!(BM, row, col)
    end
    return BM
end

function add_match{G <: Integer}(BM::BlockMatchMatrix{G}, blockrow::G, blockcol::G, row::G, col::G)
    return add_match!(copy(BM), blockrow, blockcol, row, col)
end

function add_match{G <: Integer}(BM::BlockMatchMatrix{G}, blockrow::Array{G, 1}, blockcol::Array{G, 1}, row::Array{G, 1}, col::Array{G, 1})
    return add_match!(copy(BM), blockrow, blockcol, row, col)
end

function add_match{G <: Integer}(BM::BlockMatchMatrix{G}, row::G, col::G)
    return add_match!(copy(BM), row, col)
end

function add_match{G <: Integer}(BM::BlockMatchMatrix{G}, row::Array{G, 1}, col::Array{G, 1})
    return add_match!(copy(BM), row, col)
end

"""
Return the index of the first element in A which equals val
`findindex(A, val)`
similar to indexin function but for a single element
"""
function findindex{G <: Number}(A::Array{G, 1}, val::G)
    return findfirst(x -> x == val, A)
end

"""
Perform a move on the specificed elements of BlockMatchMatrix
"""
function move_blockmatchmatrix{G <: Integer, T <: AbstractFloat}(blockrows::Array{G,1}, blockcols::Array{G,1}, BM::BlockMatchMatrix, p::T)
    newBM = copy(BM)
    rows = getrows(blockrows, newBM)
    row = StatsBase.sample(rows)
    blockrow = getblockrow(row, newBM)
    matchrows, matchcols = getmatches(blockrows[blockrows .== blockrow], blockcols[blockrows .== blockrow], newBM)
    cols = getcols(blockcols[blockrows .== blockrow], newBM)
    idx = findindex(matchrows, row)
    if idx != 0 #sampled row contains a match
        #println(1)
        col = matchcols[idx]
        blockcol = getblockcol(col, newBM)
        idx = findindex(newBM.blocks[blockrow, blockcol].rows, row - get(newBM.nrows, blockrow - 1, 0))
        deleteat!(newBM.blocks[blockrow, blockcol].rows, idx)
        deleteat!(newBM.blocks[blockrow, blockcol].cols, idx)
        if rand() < p #delete
            #println(2)
            return newBM, p / (length(cols) - length(matchcols) + 1.0)
        else #move
            #println(3)
            rowto = StatsBase.sample(push!(setdiff(getrows(blockrow, newBM), matchrows), row))
            colto = StatsBase.sample(push!(setdiff(cols, matchcols), col))
            blockcolto = getblockcol(colto, newBM)
            push!(newBM.blocks[blockrow, blockcolto].rows, rowto - get(newBM.nrows, blockrow - 1, 0))
            push!(newBM.blocks[blockrow, blockcolto].cols, colto - get(newBM.ncols, blockcolto - 1, 0))
            return newBM, 1.0
        end
    else #sampled row does not contain a match
        if length(matchcols) == length(cols) #if full move match to from different row
            #println(4)
            idx = StatsBase.sample(1:length(matchcols))
            rowfrom = matchrows[idx]
            col = matchcols[idx]
            blockrowfrom = getblockrow(rowfrom, newBM)
            blockcol = getblockcol(col, newBM)

            idx = findindex(newBM.blocks[blockrowfrom, blockcol].rows, rowfrom - get(newBM.nrows, blockrowfrom - 1, 0))
            deleteat!(newBM.blocks[blockrow, blockcol].rows, idx)
            deleteat!(newBM.blocks[blockrow, blockcol].cols, idx)
            push!(newBM.blocks[blockrow, blockcol].rows, row - get(newBM.nrows, blockrow - 1, 0))
            push!(newBM.blocks[blockrow, blockcol].cols, col - get(newBM.ncols, blockcol - 1, 0))
            return newBM, 1.0
        else #add a match
            #println(5)
            col = StatsBase.sample(setdiff(cols, matchcols))
            blockcol = getblockcol(col, BM)
            push!(newBM.blocks[blockrow, blockcol].rows, row - get(newBM.nrows, blockrow - 1, 0))
            push!(newBM.blocks[blockrow, blockcol].cols, col - get(newBM.ncols, blockcol - 1, 0))
            return newBM, p / (length(cols) - length(matchcols) + 1.0)
        end
    end
end

"""
Perform a move on the specificed elements of BlockMatchMatrix
"""
function move_blockmatchmatrix_exclude{G <: Integer, T <: AbstractFloat}(blockrows::Array{G,1}, blockcols::Array{G,1}, exrows::Array{G,1}, excols::Array{G,1}, BM::BlockMatchMatrix, p::T)
    newBM = copy(BM)
    rows = getrows(blockrows, exrows, newBM)
    row = StatsBase.sample(rows)
    blockrow = getblockrow(row, newBM)
    matchrows, matchcols = getmatches(blockrows[blockrows .== blockrow], blockcols[blockrows .== blockrow], newBM)
    cols = getcols(blockcols[blockrows .== blockrow], excols, newBM)
    idx = findindex(matchrows, row)
    if idx != 0 #sampled row contains a match
        col = matchcols[idx]
        blockcol = getblockcol(col, newBM)
        idx = findindex(newBM.blocks[blockrow, blockcol].rows, row - get(newBM.nrows, blockrow - 1, 0))
        deleteat!(newBM.blocks[blockrow, blockcol].rows, idx)
        deleteat!(newBM.blocks[blockrow, blockcol].cols, idx)
        if rand() < p #delete
            return newBM, p / (length(cols) - length(matchcols) + 1)
        else #move
            rowto = StatsBase.sample(push!(setdiff(getrows(blockrow, exrows, newBM), matchrows), row))
            colto = StatsBase.sample(push!(setdiff(cols, matchcols), col))
            blockcolto = getblockcol(colto, newBM)
            push!(newBM.blocks[blockrow, blockcolto].rows, rowto - get(newBM.nrows, blockrow - 1, 0))
            push!(newBM.blocks[blockrow, blockcolto].cols, colto - get(newBM.ncols, blockcolto - 1, 0))
            return newBM, 1.0
        end
    else #sampled row does not contain a match
        if length(matchcols) == length(cols) #if full move match to from different row
            idx = StatsBase.sample(1:length(matchcols))
            rowfrom = matchrows[idx]
            col = matchcols[idx]
            blockrowfrom = getblockrow(rowfrom, newBM)
            blockcol = getblockcol(col, newBM)

            idx = findindex(newBM.blocks[blockrowfrom, blockcol].rows, rowfrom - get(newBM.nrows, blockrowfrom - 1, 0))
            deleteat!(newBM.blocks[blockrow, blockcol].rows, idx)
            deleteat!(newBM.blocks[blockrow, blockcol].cols, idx)
            push!(newBM.blocks[blockrow, blockcol].rows, row - get(newBM.nrows, blockrow - 1, 0))
            push!(newBM.blocks[blockrow, blockcol].cols, col - get(newBM.ncols, blockcol - 1, 0))
            
            return newBM, 1.0
        else #add a match
            col = StatsBase.sample(setdiff(cols, matchcols))
            blockcol = getblockcol(col, BM)
            push!(newBM.blocks[blockrow, blockcol].rows, row - get(newBM.nrows, blockrow - 1, 0))
            push!(newBM.blocks[blockrow, blockcol].cols, col - get(newBM.ncols, blockcol - 1, 0))
            return newBM, p / (length(cols) - length(matchcols) + 1)
        end
    end
end

function =={G <: BlockMatchMatrix}(BM1::G, BM2::G)
    return all(BM1.blocks .== BM2.blocks)
end

#get(cumsum([1,6,3,5]), 1, 0)
#enumerate()iterator that yields (i, x) where i is an index starting at 1, and x is the ith value from the given iterator
#indicies
#eachindex

# 0.5 * (erf(1. / sqrt(2)) - erf(0. / sqrt(2)))
#Distributions.cdf(Distributions.Normal(), 1) - Distributions.cdf(Distributions.Normal(), 0)
