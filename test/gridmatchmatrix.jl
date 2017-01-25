import StatsBase
using Base.Test

include("../src/matching_matrix.jl")
include("../src/gridmatchingmatrix.jl")

M = MatchMatrix(2,3)
copy(M)

GM1 = GridMatchMatrix([2,2], [2,3])
@test GM1.nrows == [2,2]
@test GM1.ncols == [2,3]
@test GM1.cumrows == [2,4]
@test GM1.cumcols == [2,5]

GM2 = copy(GM1)

GM1.grid[1,1].rows = [1, 2]
GM1.grid[1,1].cols = [1, 2]
GM1.grid[2,2].rows = [1]
GM1.grid[2,2].cols = [3]

@test getgridrow(1, GM1) == 1
@test getgridrow(3, GM1) == 2
@test getgridcol(2, GM1) == 1
@test getgridcol(5, GM1) == 2

@test getrows([1, 2], GM1) == collect(1:4)
@test getrows(1, GM1) == collect(1:2)
@test getcols([1, 2], GM1) == collect(1:5)
@test getcols(2, GM1) == collect(3:5)

for (row, col) in getindicies([1, 2], [1, 2], GM1)
    println(row, col)
end


rows = Array{Float64}(0)
cols = Array{Float64}(0)
for (ii, jj) in zip(1:2, 1:2)
    append!(rows, (get(GM1.cumrows, ii - 1, 0) + 1):GM1.cumrows[ii])
    append!(cols, (get(GM1.cumcols, jj - 1, 0) + 1):GM1.cumcols[jj])
end

getmatches([1,1,2,2], [1,2,1,2], GM1)
getmatches([1,2], [1,2], GM1)

GM.grid[1,1]
GM.grid[1,2]
GM.grid[2,1]
GM.grid[2,2]
getmatches([1,1,2,2], [1,2,1,2], GM)

GM = copy(GM1)
@time for ii in 1:100
    GM, p = move_matchgrid([1,2], [1, 2], GM1, 0.5)
end
