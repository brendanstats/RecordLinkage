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
    blocks = Array{MatchMatrix{G}}(length(nrows), length(ncols))
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
function Base.copy{G <: Integer}(BM::BlockMatchMatrix{G})
    newblocks = Array{MatchMatrix{G}}(length(BM.nrows), length(BM.ncols))
    for ii in eachindex(newblocks)
        newblocks[ii] = copy(BM.blocks[ii])
    end
    return BlockMatchMatrix(newblocks, copy(BM.nrows), copy(BM.ncols), copy(BM.cumrows), copy(BM.cumcols), copy(BM.nrow), copy(BM.ncol))
end

"""
Return element type of BlockMatchMatrix
"""
Base.eltype{G <: Integer}(BM::BlockMatchMatrix{G}) = G

"""
Return the row of the grid of MatchMatrix corresponding to supplied row index of full matrix

`getblockrow(row, BM)`
"""
function getblockrow{G <: Integer}(row::G, BM::BlockMatchMatrix{G})
    return searchsortedfirst(BM.cumrows, row)
end

"""
Return the column of the grid of MatchMatrix corresponding to supplied column index of full matrix
"""
function getblockcol{G <: Integer}(col::G, BM::BlockMatchMatrix{G})
    return searchsortedfirst(BM.cumcols, col)
end

"""
Return the index of the row from within a block, so if block contains rows 20 to 40 then return would be 1 for 20.
"""
function getrowofblock{G <: Integer}(row::G, blockrow::G, BM::BlockMatchMatrix{G})
    return row - get(BM.cumrows, blockrow - one(G), zero(G))
end

function getrowofblock{G <: Integer}(row::G, BM::BlockMatchMatrix{G})
    return row - get(BM.cumrows, getblockrow(row, BM) - one(G), zero(G))
end

"""
Return the index of the row from within a block, so if block contains rows 20 to 40 then return would be 1 for 20.
"""
function getcolofblock{G <: Integer}(col::G, blockcol::G, BM::BlockMatchMatrix{G})
    return col - get(BM.cumcols, blockcol - one(G), zero(G))
end

function getcolofblock{G <: Integer}(col::G, BM::BlockMatchMatrix{G})
    return col - get(BM.cumcols, getblockcol(col, BM) - one(G), zero(G))
end

"""
Return the rows of the full matrix corresponding to the given indicies
"""
function getrows{G <: Integer}(blockrows::Array{G,1}, BM::BlockMatchMatrix{G})
    rows = Array{G}(0)
    for ii in unique(blockrows)
        append!(rows, (get(BM.cumrows, ii - 1, 0) + 1):BM.cumrows[ii])
    end
    return rows
end

function getrows{G <: Integer}(blockrow::G, BM::BlockMatchMatrix{G})
    return collect((get(BM.cumrows, blockrow - 1, 0) + 1):BM.cumrows[blockrow])
end

"""
Return the columns of the full matrix corresponding to the given indicies
"""
function getcols{G <: Integer}(blockcols::Array{G,1}, BM::BlockMatchMatrix{G})
    cols = Array{G}(0)
    for ii in unique(blockcols)
        append!(cols, (get(BM.cumcols, ii - 1, 0) + 1):BM.cumcols[ii])
    end
    return cols
end

function getcols{G <: Integer}(blockcol::G, BM::BlockMatchMatrix{G})
    return collect((get(BM.cumcols, blockcol - 1, 0) + 1):BM.cumcols[blockcol])
end

"""
Return the rows of the full matrix corresponding to the given indicies, excluding listed entries
"""
function getrows{G <: Integer}(blockrows::Array{G, 1}, exrows::Array{G, 1}, BM::BlockMatchMatrix{G})
    rows = Array{G}(0)
    for ii in unique(blockrows)
        append!(rows, (get(BM.cumrows, ii - 1, 0) + 1):BM.cumrows[ii])
    end
    return setdiff(rows, exrows)
end

function getrows{G <: Integer}(blockrow::G, exrows::Array{G, 1}, BM::BlockMatchMatrix{G})
    return setdiff(collect((get(BM.cumrows, blockrow - 1, 0) + 1):BM.cumrows[blockrow]), exrows)
end

"""
Return the columns of the full matrix corresponding to the given indicies, excluding listed entries
"""
function getcols{G <: Integer}(blockcols::Array{G,1}, excols::Array{G, 1}, BM::BlockMatchMatrix{G})
    cols = Array{G}(0)
    for ii in unique(blockcols)
        append!(cols, (get(BM.cumcols, ii - 1, 0) + 1):BM.cumcols[ii])
    end
    return setdiff(cols, excols)
end

function getcols{G <: Integer}(blockcol::G, excols::Array{G, 1}, BM::BlockMatchMatrix{G})
    return setdiff(collect((get(BM.cumcols, blockcol - 1, 0) + 1):BM.cumcols[blockcol]), excols)
end


"""
Return the rows and columns of the full matrix corresponding to the given indicies of grid of blocks
"""
function getindicies{G <: Integer}(blockrows::Array{G,1}, blockcols::Array{G,1}, BM::BlockMatchMatrix{G})
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
function getmatches{G <: Integer}(blockrow::G, blockcol::G, BM::BlockMatchMatrix{G})
    rows = BM.blocks[blockrow, blockcol].rows .+ get(BM.cumrows, blockrow - 1, 0)
    cols = BM.blocks[blockrow, blockcol].cols .+ get(BM.cumcols, blockcol - 1, 0)
    return rows, cols
end

function getmatches{G <: Integer}(blockrows::Array{G,1}, blockcols::Array{G,1}, BM::BlockMatchMatrix{G})
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

function getmatches{G <: Integer}(BM::BlockMatchMatrix{G})
    rows = Array{G}(0)
    cols = Array{G}(0)
    for jj in 1:size(BM.blocks, 2)
        for ii in 1:size(BM.blocks, 1)
            if length(BM.blocks[ii, jj].rows) > 0
                append!(rows, BM.blocks[ii, jj].rows .+ get(BM.cumrows, ii - 1, 0))
                append!(cols, BM.blocks[ii, jj].cols .+ get(BM.cumcols, jj - 1, 0))
            end
        end
    end
    return rows, cols
end

"""
Return a bit array the length of the total nrow with elements true if the row is empty and false if it is full
"""
function getrowempty{G <: Integer}(BM::BlockMatchMatrix{G})
    out = trues(BM.nrow)
    for ii in 1:length(BM.cumrows)
        rowadj = get(BM.cumrows, ii - one(G), zero(G))
        for jj in 1:length(BM.cumcols)
            out[BM.blocks[ii, jj].rows .+ rowadj] .= false
        end
    end
    return out
end

function getrowempty{G <: Integer}(blockrows::Array{G, 1}, BM::BlockMatchMatrix{G})
    out = falses(BM.nrow)
    out[getrows(blockrows, BM)] .= true
    for ii in blockrows
        rowadj = get(BM.cumrows, ii - one(G), zero(G))
        for jj in 1:length(BM.cumcols)
            out[BM.blocks[ii, jj].rows .+ rowadj] .= false
        end
    end
    return out
end

function getrowempty{G <: Integer}(blockrows::Array{G, 1}, exrows::Array{G, 1}, BM::BlockMatchMatrix{G})
    out = getrowempty(blockrows, BM)
    out[exrows] .= false
    return out
end

"""
Return a bit array the length of the total ncol with elements true if the col is empty and false if it is full
"""
function getcolempty{G <: Integer}(BM::BlockMatchMatrix{G})
    out = trues(BM.ncol)
    for jj in 1:length(BM.cumcols)
        coladj = get(BM.cumcols, jj - one(G), zero(G))
        for ii in 1:length(BM.cumrows)
            out[BM.blocks[ii, jj].cols .+ coladj] .= false
        end
    end
    return out
end

function getcolempty{G <: Integer}(blockcols::Array{G, 1}, BM::BlockMatchMatrix{G})
    out = falses(BM.ncol)
    out[getcols(blockcols, BM)] .= true
    for jj in blockcols
        coladj = get(BM.cumcols, jj - one(G), zero(G))
        for ii in 1:length(BM.cumrows)
            out[BM.blocks[ii, jj].cols .+ coladj] .= false
        end
    end
    return out
end

function getcolempty{G <: Integer}(blockcols::Array{G, 1}, excols::Array{G, 1}, BM::BlockMatchMatrix{G})
    out = getcolempty(blockcols, BM)
    out[excols] .= false
    return out
end


"""
Return all of the empty rows, used for adding matches / moving, wrapper around getrowempty
"""
function getemptyrows{G <: Integer}(BM::BlockMatchMatrix{G})
    return find(getrowempty(BM))
end

function getemptyrows{G <: Integer}(blockrows::Array{G, 1}, BM::BlockMatchMatrix{G})
    return find(getrowempty(blockrows, BM))
end

function getemptyrows{G <: Integer}(blockrows::Array{G, 1}, exrows::Array{G, 1}, BM::BlockMatchMatrix{G})
    return find(getrowempty(blockrows, exrows, BM))
end

"""
Return all of the empty columns, used for adding matches / moving, wrapper around getcolempty
"""
function getemptycols{G <: Integer}(BM::BlockMatchMatrix{G})
    return find(getcolempty(BM))
end

function getemptycols{G <: Integer}(blockcols::Array{G, 1}, BM::BlockMatchMatrix{G})
    return find(getcolempty(blockcols, BM))
end

function getemptycols{G <: Integer}(blockcols::Array{G, 1}, excols::Array{G, 1}, BM::BlockMatchMatrix{G})
    return find(getcolempty(blockcols, excols, BM))
end

"""
Get number of links in each block returned as a matrix with the same dimensions as the BlockMatchMatrix
"""
function getblocknlinks{G <: Integer}(BM::BlockMatchMatrix{G})
    out = zeros(G, size(BM.blocks)...)
    for ii in eachindex(BM.blocks)
        out[ii] = length(BM.blocks[ii].rows)
    end
    return out
end

"""
Reduce entries of nrows by the number of elements in each block contained in exrows
`getnrowseffective(BM, exrows)`
"""
function getnrowsremaining{G <: Integer}(BM::BlockMatchMatrix{G}, exrows::Array{G, 1})
    nrowsRemain = copy(BM.nrows)
    for row in exrows
        nrowsRemain[getblockrow(row, BM)] -= one(G)
    end
    return nrowsRemain
end

"""
Reduce entries of ncols by the number of elements in each block contained in excols
`getncolseffective(BM, exrows)`
"""
function getncolsremaining{G <: Integer}(BM::BlockMatchMatrix{G}, excols::Array{G, 1})
    ncolsRemain = copy(BM.ncols)
    for col in excols
        ncolsRemain[getblockcol(col, BM)] -= one(G)
    end
    return ncolsRemain
end

"""
Add a match to a BlockMatchMatrix
"""
function add_match!{G <: Integer}(BM::BlockMatchMatrix{G}, blockrow::G, blockcol::G, row::G, col::G)
    rowadj = row - get(BM.cumrows, blockrow - 1, 0)
    coladj = col - get(BM.cumcols, blockcol - 1, 0)
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
Remove a match from the appropriate block, row and column index provided should be global not block specific
"""
function remove_match!{G <: Integer}(BM::BlockMatchMatrix{G}, row::G, col::G, blockrow::G, blockcol::G)
    rob = getrowofblock(row, blockrow, BM)
    cob = getcolofblock(col, blockcol, BM)

    remove_match!(BM.blocks[blockrow, blockcol], rob, cob)
    return BM
end

function remove_match!{G <: Integer}(BM::BlockMatchMatrix{G}, row::G, col::G)
    blockrow = getblockrow(row, BM)
    blockcol = getblockcol(col, BM)
    return remove_match!(BM, row, col, blockrow, blockcol)
end

function =={G <: BlockMatchMatrix}(BM1::G, BM2::G)
    return all(BM1.blocks .== BM2.blocks)
end

"""
Return the index of the first element in A which equals val
`findindex(A, val)`
similar to indexin function but for a single element, also consider searchsorted
"""
function findindex{G <: Number}(A::Array{G, 1}, val::G)
    return findfirst(x -> x == val, A)
end

"""
Perform a move on the specificed elements of BlockMatchMatrix
"""
function move_blockmatchmatrix{G <: Integer, T <: AbstractFloat}(blockrows::Array{G,1}, blockcols::Array{G,1}, BM::BlockMatchMatrix{G}, p::T)
    newBM = copy(BM)

    #Select row to add / delete / move match to
    rows = getrows(blockrows, newBM)
    row = StatsBase.sample(rows)
    blockrow = getblockrow(row, newBM)

    #Determine if selected row is currently matched
    blockidx = blockrows .== blockrow
    matchrows, matchcols = getmatches(blockrows[blockidx], blockcols[blockidx], newBM)
    cols = getcols(blockcols[blockidx], newBM)
    idx = findindex(matchrows, row)

    if idx != 0 #sampled row contains a match

        col = matchcols[idx]
        blockcol = getblockcol(col, newBM)
        remove_match!(newBM, row, col, blockrow, blockcol)
        
        if rand() < p #don't add back

            #column count might need to be adjusted
            #println("Delete")
            return newBM, p / (length(cols) - length(matchcols) + 1.0)

        else #move

            #"move" to same row allowed since row deleted before checking availability
            #emptycols = getemptycols(blockcols[blockidx], newBM)
            rowto = StatsBase.sample(getemptyrows(blockrows[blockidx], newBM))
            add_match!(newBM, rowto, col)
            #println("Delete - Move")
            return newBM, 1.0
            
        end

    else #sampled row does not contain a match
        emptycols = getemptycols(blockcols[blockidx], newBM)

        #if full move match to from different row
        if length(emptycols) == 0

            col = StatsBase.sample(cols)
            idx = findindex(matchcols, col)

            if idx == 0
                error("no empty cols and selected column not a match check initial matrix for consistency with selected blocks")
            end

            remove_match!(newBM, matchrows[idx], col)
            add_match!(newBM, row, col)

            #println("Add - Move")
            return newBM, 1.0
        else #add a match
            
            col = StatsBase.sample(emptycols)
            add_match!(newBM, row, col)

            #println("Add")
            return newBM, p / (length(cols) - length(matchcols) + 1.0)
        end
    end
end

"""
Perform a move on the specificed elements of BlockMatchMatrix
"""
function move_blockmatchmatrix_exclude{G <: Integer, T <: AbstractFloat}(blockrows::Array{G,1}, blockcols::Array{G,1}, exrows::Array{G,1}, excols::Array{G,1}, BM::BlockMatchMatrix{G}, p::T)
    newBM = copy(BM)

    #all columns or all rows contains a match from the previous stage
    if length(exrows) == BM.nrows || length(excols) == BM.ncols
        return newBM, 1.0
    end

    #Select row to add / delete / move match to
    rows = getrows(blockrows, exrows, newBM)
    row = StatsBase.sample(rows)
    blockrow = getblockrow(row, newBM)

    #matchrows, matchcols = getmatches(blockrows[blockidx], blockcols[blockidx], newBM)
    matchrows, matchcols = getmatches(blockrows, blockcols, newBM)
    idx = findindex(matchrows, row)
    
    #Determine if selected row is currently matched
    blockidx = blockrows .== blockrow
    cols = getcols(blockcols[blockidx], excols, newBM)
    
    #all columns in blocks overlapping with row matched in a previous stage
    if length(cols) == 0
        return newBM, 1.0
    end
    
    if idx != 0 #sampled row contains a match
        
        col = matchcols[idx]
        blockcol = getblockcol(col, newBM)
        remove_match!(newBM, row, col, blockrow, blockcol)
        
        if rand() < p #delete
            
            return newBM, p / (length(cols) - length(matchcols) + 1)
            
        else #move

            #"move" to same row allowed since row deleted before checking availability
            #emptycols = getemptycols(blockcols[blockidx], newBM)
            #rowto = StatsBase.sample(getemptyrows(blockrows[blockidx], exrows, newBM)) #integer division error
            emptyrows = getemptyrows([blockrow], exrows, newBM)
            if length(emptyrows) == 0
                println("row: ", row)
                println("col: ", col)
                println("blockrow: ", blockrow)
                println("blockcol: ", blockcol)
                println("matchrows: ", matchrows)
                println("matchcols: ", matchcols)
                println("exrows: ", exrows)
                println("excols: ", excols)
                println("nrows: ", newBM.nrows)
                println("ncols: ", newBM.ncols)
                println(BM.blocks[blockrow, blockcol])
                println(newBM.blocks[blockrow, blockcol])
                println(BM)
                error("no empty rows after deletion, logic error")
            end
            rowto = StatsBase.sample(emptyrows) #integer division error
            add_match!(newBM, rowto, col)
            return newBM, 1.0
            
        end
    else #sampled row does not contain a match
        emptycols = getemptycols(blockcols[blockidx], excols, newBM)

        #if full move match to from different row
        if length(emptycols) == 0

            col = StatsBase.sample(cols)
            idx = findindex(matchcols, col)
            
            if idx == 0
                blockcol = getblockcol(col, newBM)
                println("row: ", row)
                println("col: ", col)
                println("blockrow: ", blockrow)
                println("blockcol: ", blockcol)
                println("matchrows: ", matchrows)
                println("matchcols: ", matchcols)
                println("exrows: ", exrows)
                println("excols: ", excols)
                println("nrows: ", newBM.nrows)
                println("ncols: ", newBM.ncols)
                println(BM.blocks[blockrow, blockcol])
                println(newBM.blocks[blockrow, blockcol])
                println(BM)

                error("no empty cols and selected column not a match check initial matrix for consistency with selected blocks")
                #println("no empty cols and selected column not a match check initial matrix for consistency with selected blocks")
                return newBM, 1.0
            end
            
            remove_match!(newBM, matchrows[idx], col)
            add_match!(newBM, row, col)
            
            return newBM, 1.0
        else #add a match

            col = StatsBase.sample(emptycols)
            add_match!(newBM, row, col)
            
            return newBM, p / (length(cols) - length(matchcols) + 1.0)
        end
    end
end
