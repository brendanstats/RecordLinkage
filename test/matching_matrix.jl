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
@test_throws(ErrorException, in(M, 1, 7))

@test findindex(M, 1, 2) == 1
@test findindex(M, 4, 3) == 2
@test findindex(M, 4, 4) == 0

add_match!(M, 3, 5)
@test M.rows == [1, 4, 3] && M.cols == [2, 3, 5]
add_match!(M, 3, 5, check = true)
@test M.rows == [1, 4, 3] && M.cols == [2, 3, 5]

remove_match!(M, 1, 2)
@test M.rows == [4, 3] && M.cols == [3, 5]

@test convert(Array{Int64, 2}, M) == [0 0 0 0 0;
                                      0 0 0 0 0;
                                      0 0 0 0 1;
                                      0 0 1 0 0;
                                      0 0 0 0 0]
@test empty_rows(M) == [1, 2, 5]
@test empty_cols(M) == [1, 2, 4]

#Should add some unit tests with specific matricies

P = move_matchmatrix(M, 0.5)
ratio_pmove_row(M, P, 0.4)
ratio_pmove_row(P, M, 0.4)
ratio_pmove_col(M, P, 0.5)
ratio_pmove_col(P, M, 0.5)
ratio_pmove(M, P, 0.4)
ratio_pmove(P, M, 0.4)
