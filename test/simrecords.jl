using Base.Test

import StatsBase
include("../src/simrecords.jl")

pM = [0.9, 0.8]
pU = [0.2, 0.3]
npop = 10
na = 3
nb = 6
obs, C =  simulate_singlelinkage_binary(npop, na, nb, pM, pU)

@testset "Input Tests" begin
    @test_throws(ErrorException, simulate_singlelinkage_binary(npop, na, npop + 3, pM, pU))
    @test_throws(ErrorException, simulate_singlelinkage_binary(npop, npop + 2, nb, pM, [0.2]))
    @test_throws(ErrorException, simulate_singlelinkage_binary(npop, na, nb, pM, [pU; 0.25]))
    @test_throws(ErrorException, simulate_singlelinkage_binary(npop, na, nb, [1.01, 0.8], pU))
end

@testset "Output Tests" begin
    @test isa(obs, Array{Int64, 3})
    @test size(obs) == (length(pM), na, nb)
    @test isa(C, MatchMatrix)
    @test C.nrow == na
    @test C.ncol == nb
end
