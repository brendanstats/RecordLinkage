"""
Type to hold an Array with MatchMatrix elements recording other information, intended to be used for blocking
"""
type GridMatchMatrix{G}
    grid::Array{MatchMatrix, 2}
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
function GridMatchMatrix{G <: Integer}(nrows::Array{G, 1}, ncols::Array{G, 1})
    grid = Array{MatchMatrix}(length(nrows), length(ncols))
    for ii in eachindex(nrows)
        for jj in eachindex(ncols)
            grid[ii, jj] = MatchMatrix(nrows[ii], ncols[jj])
        end
    end
    return GridMatchMatrix(grid, nrows, ncols, cumsum(nrows), cumsum(ncols), sum(nrows), sum(ncols))
end

"""
Constructor based only on block size
"""
function GridMatchMatrix{G <: Integer}(nrows::Array{G, 1}, ncols::Array{G, 1}, M::MatchMatrix)
    GM = GridMatchMatrix(nrows, ncols)
    for ii in 1:length(M.cols)
        grow = getgridrow(M.rows[ii], GM)
        gcol = getgridcol(M.cols[ii], GM)
        push!(GM.grid[grow, gcol].rows, M.rows[ii] - get(GM.cumrows, grow - 1, 0))
        push!(GM.grid[grow, gcol].cols, M.cols[ii] - get(GM.cumcols, gcol - 1, 0))
    end
    return GM
end

"""
Constructor based on Array with type of MatchMatrix, row elements  must all be same height and column elements must all be same width
"""
function GridMatchMatrix{G <: Integer}(grid::Array{MatchMatrix{G}, 2})
    n, m = size(grid)
    nrows = Array{G}(n)
    ncols = Array{G}(m)
    #get row dimensions
    for ii in 1:n
        nrows[ii] = grid[ii, 1].nrow
    end
    #get col dimensions
    for jj in 1:m
        ncols[jj] = grid[1, jj].ncol
    end
    for ii in 1:n
        for jj in 1:m
            if nrows[ii] != grid[ii, jj].nrow
                error("dimensions of match matricies do not match within row")
            end
            if ncols[jj] != grid[ii, jj].ncol
                error("dimensions of match matricies do not match within col")
            end
        end
    end
    return GridMatchMatrix(grid, nrows, ncols, cumsum(nrows), cumsum(ncols), sum(nrows), sum(ncols))
end

"""
Create a shallow copy of GridMatchMatrix
"""
function Base.copy(GM::GridMatchMatrix)
    newgrid = Array{MatchMatrix}(length(GM.nrows), length(GM.ncols))
    for ii in eachindex(newgrid)
        newgrid[ii] = copy(GM.grid[ii])
    end
    return GridMatchMatrix(newgrid, copy(GM.nrows), copy(GM.ncols), copy(GM.cumrows), copy(GM.cumcols), copy(GM.nrow), copy(GM.ncol))
end

"""
Return the row of the grid of MatchMatrix corresponding to supplied row index of full matrix

`getgridrow(row, GM)`
"""
function getgridrow{G <: Integer}(row::G, GM::GridMatchMatrix)
    return searchsortedfirst(GM.cumrows, row)
end

"""
Return the column of the grid of MatchMatrix corresponding to supplied column index of full matrix
"""
function getgridcol{G <: Integer}(col::G, GM::GridMatchMatrix)
    return searchsortedfirst(GM.cumcols, col)
end

"""
Return the rows of the full matrix corresponding to the given indicies
"""
function getrows{G <: Integer}(grows::Array{G,1}, GM::GridMatchMatrix)
    rows = Array{G}(0)
    for ii in unique(grows)
        append!(rows, (get(GM.cumrows, ii - 1, 0) + 1):GM.cumrows[ii])
    end
    return rows
end

function getrows{G <: Integer}(grow::G, GM::GridMatchMatrix)
    return collect((get(GM.cumrows, grow - 1, 0) + 1):GM.cumrows[grow])
end

"""
Return the columns of the full matrix corresponding to the given indicies
"""
function getcols{G <: Integer}(gcols::Array{G,1}, GM::GridMatchMatrix)
    cols = Array{G}(0)
    for ii in unique(gcols)
        append!(cols, (get(GM.cumcols, ii - 1, 0) + 1):GM.cumcols[ii])
    end
    return cols
end

function getcols{G <: Integer}(gcol::G, GM::GridMatchMatrix)
    return collect((get(GM.cumcols, gcol - 1, 0) + 1):GM.cumcols[gcol])
end

"""
Return the rows of the full matrix corresponding to the given indicies, excluding listed entries
"""
function getrows{G <: Integer}(grows::Array{G, 1}, exrows::Array{G, 1}, GM::GridMatchMatrix)
    rows = Array{G}(0)
    for ii in unique(grows)
        append!(rows, (get(GM.cumrows, ii - 1, 0) + 1):GM.cumrows[ii])
    end
    return setdiff(rows, exrows)
end

function getrows{G <: Integer}(grow::G, exrows::Array{G, 1}, GM::GridMatchMatrix)
    return setdiff(collect((get(GM.cumrows, grow - 1, 0) + 1):GM.cumrows[grow]), exrows)
end

"""
Return the columns of the full matrix corresponding to the given indicies, excluding listed entries
"""
function getcols{G <: Integer}(gcols::Array{G,1}, excols::Array{G, 1}, GM::GridMatchMatrix)
    cols = Array{G}(0)
    for ii in unique(gcols)
        append!(cols, (get(GM.cumcols, ii - 1, 0) + 1):GM.cumcols[ii])
    end
    return setdiff(cols, excols)
end

function getcols{G <: Integer}(gcol::G, excols::Array{G, 1}, GM::GridMatchMatrix)
    return setdiff(collect((get(GM.cumcols, gcol - 1, 0) + 1):GM.cumcols[gcol]), excols)
end


"""
Return the rows and columns of the full matrix corresponding to the given indicies of grid of blocks
"""
function getindicies{G <: Integer}(grows::Array{G,1}, gcols::Array{G,1}, GM::GridMatchMatrix)
    rows = Array{G}(0)
    cols = Array{G}(0)
    for (ii, jj) in zip(grows, gcols)
        append!(rows, (get(GM.cumrows, ii - 1, 0) + 1):GM.cumrows[ii])
        append!(cols, (get(GM.cumcols, jj - 1, 0) + 1):GM.cumcols[jj])
    end
    return zip(rows, cols)
end

"""
Get columns and rows containing a match
"""
function getmatches{G <: Integer}(grows::Array{G,1}, gcols::Array{G,1}, GM::GridMatchMatrix)
    rows = Array{G}(0)
    cols = Array{G}(0)
    for (ii, jj) in zip(grows, gcols)
        if length(GM.grid[ii, jj].rows) > 0
            append!(rows, GM.grid[ii, jj].rows .+ get(GM.cumrows, ii - 1, 0))
            append!(cols, GM.grid[ii, jj].cols .+ get(GM.cumcols, jj - 1, 0))
        end
    end
    return rows, cols
end

function getmatches(GM::GridMatchMatrix)
    rows = Array{eltype(GM.grid[1,1].rows)}(0)
    cols = Array{eltype(GM.grid[1,1].cols)}(0)
    for ii in 1:size(GM.grid, 1)
        for jj in 1:size(GM.grid, 2)
            if length(GM.grid[ii, jj].rows) > 0
                append!(rows, GM.grid[ii, jj].rows .+ get(GM.cumrows, ii - 1, 0))
                append!(cols, GM.grid[ii, jj].cols .+ get(GM.cumcols, jj - 1, 0))
            end
        end
    end
    return rows, cols
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
Perform a move on the specificed elements of GridMatchMatrix
"""
function move_gridmatchmatrix{G <: Integer, T <: AbstractFloat}(grows::Array{G,1}, gcols::Array{G,1}, GM::GridMatchMatrix, p::T)
    newGM = copy(GM)
    rows = getrows(grows, newGM)
    row = StatsBase.sample(rows)
    grow = getgridrow(row, newGM)
    matchrows, matchcols = getmatches(grows[grows .== grow], gcols[grows .== grow], newGM)
    cols = getcols(gcols[grows .== grow], newGM)
    idx = findindex(matchrows, row)
    if idx != 0 #sampled row contains a match
        #println(1)
        col = matchcols[idx]
        gcol = getgridcol(col, newGM)
        idx = findindex(newGM.grid[grow, gcol].rows, row - get(newGM.nrows, grow - 1, 0))
        deleteat!(newGM.grid[grow, gcol].rows, idx)
        deleteat!(newGM.grid[grow, gcol].cols, idx)
        if rand() < p #delete
            #println(2)
            return newGM, p / (length(cols) - length(matchcols) + 1.0)
        else #move
            #println(3)
            rowto = StatsBase.sample(push!(setdiff(getrows(grow, newGM), matchrows), row))
            colto = StatsBase.sample(push!(setdiff(cols, matchcols), col))
            gcolto = getgridcol(colto, newGM)
            push!(newGM.grid[grow, gcolto].rows, rowto - get(newGM.nrows, grow - 1, 0))
            push!(newGM.grid[grow, gcolto].cols, colto - get(newGM.ncols, gcolto - 1, 0))
            return newGM, 1.0
        end
    else #sampled row does not contain a match
        if length(matchcols) == length(cols) #if full move match to from different row
            #println(4)
            idx = StatsBase.sample(1:length(matchcols))
            rowfrom = matchrows[idx]
            col = matchcols[idx]
            growfrom = getgridrow(rowfrom, newGM)
            gcol = getgridcol(col, newGM)

            idx = findindex(newGM.grid[growfrom, gcol].rows, rowfrom - get(newGM.nrows, growfrom - 1, 0))
            deleteat!(newGM.grid[grow, gcol].rows, idx)
            deleteat!(newGM.grid[grow, gcol].cols, idx)
            push!(newGM.grid[grow, gcol].rows, row - get(newGM.nrows, grow - 1, 0))
            push!(newGM.grid[grow, gcol].cols, col - get(newGM.ncols, gcol - 1, 0))
            return newGM, 1.0
        else #add a match
            #println(5)
            col = StatsBase.sample(setdiff(cols, matchcols))
            gcol = getgridcol(col, GM)
            push!(newGM.grid[grow, gcol].rows, row - get(newGM.nrows, grow - 1, 0))
            push!(newGM.grid[grow, gcol].cols, col - get(newGM.ncols, gcol - 1, 0))
            return newGM, p / (length(cols) - length(matchcols) + 1.0)
        end
    end
end

"""
Perform a move on the specificed elements of GridMatchMatrix
"""
function move_gridmatchmatrix_exclude{G <: Integer, T <: AbstractFloat}(grows::Array{G,1}, gcols::Array{G,1}, exrows::Array{G,1}, excols::Array{G,1}, GM::GridMatchMatrix, p::T)
    newGM = copy(GM)
    rows = getrows(grows, exrows, newGM)
    row = StatsBase.sample(rows)
    grow = getgridrow(row, newGM)
    matchrows, matchcols = getmatches(grows[grows .== grow], gcols[grows .== grow], newGM)
    cols = getcols(gcols[grows .== grow], excols, newGM)
    idx = findindex(matchrows, row)
    if idx != 0 #sampled row contains a match
        col = matchcols[idx]
        gcol = getgridcol(col, newGM)
        idx = findindex(newGM.grid[grow, gcol].rows, row - get(newGM.nrows, grow - 1, 0))
        deleteat!(newGM.grid[grow, gcol].rows, idx)
        deleteat!(newGM.grid[grow, gcol].cols, idx)
        if rand() < p #delete
            return newGM, p / (length(cols) - length(matchcols) + 1)
        else #move
            rowto = StatsBase.sample(push!(setdiff(getrows(grow, exrows, newGM), matchrows), row))
            colto = StatsBase.sample(push!(setdiff(cols, matchcols), col))
            gcolto = getgridcol(colto, newGM)
            push!(newGM.grid[grow, gcolto].rows, rowto - get(newGM.nrows, grow - 1, 0))
            push!(newGM.grid[grow, gcolto].cols, colto - get(newGM.ncols, gcolto - 1, 0))
            return newGM, 1.0
        end
    else #sampled row does not contain a match
        if length(matchcols) == length(cols) #if full move match to from different row
            idx = StatsBase.sample(1:length(matchcols))
            rowfrom = matchrows[idx]
            col = matchcols[idx]
            growfrom = getgridrow(rowfrom, newGM)
            gcol = getgridcol(col, newGM)

            idx = findindex(newGM.grid[growfrom, gcol].rows, rowfrom - get(newGM.nrows, growfrom - 1, 0))
            deleteat!(newGM.grid[grow, gcol].rows, idx)
            deleteat!(newGM.grid[grow, gcol].cols, idx)
            push!(newGM.grid[grow, gcol].rows, row - get(newGM.nrows, grow - 1, 0))
            push!(newGM.grid[grow, gcol].cols, col - get(newGM.ncols, gcol - 1, 0))
            
            return newGM, 1.0
        else #add a match
            col = StatsBase.sample(setdiff(cols, matchcols))
            gcol = getgridcol(col, GM)
            push!(newGM.grid[grow, gcol].rows, row - get(newGM.nrows, grow - 1, 0))
            push!(newGM.grid[grow, gcol].cols, col - get(newGM.ncols, gcol - 1, 0))
            return newGM, p / (length(cols) - length(matchcols) + 1)
        end
    end
end

function =={G <: GridMatchMatrix}(GM1::G, GM2::G)
    return all(GM1.grid .== GM2.grid)
end

#get(cumsum([1,6,3,5]), 1, 0)
#enumerate()iterator that yields (i, x) where i is an index starting at 1, and x is the ith value from the given iterator
#indicies
#eachindex

# 0.5 * (erf(1. / sqrt(2)) - erf(0. / sqrt(2)))
#Distributions.cdf(Distributions.Normal(), 1) - Distributions.cdf(Distributions.Normal(), 0)
