import StatsBase
import Distributions
using Base.Test

include("../src/matching_matrix.jl")

#@test
#@test_throws ExceptionError
M = MatchMatrix([1,4], [2,3], 5, 5)
@test in(M, 1, 2) == true
@test in(M, 1, 3) == false
@test in(M, 1, 4) == false
@test in(M, 4, 3) == true
@test_throws in(M, 1, 7)

findindex(M, 4, 3)
