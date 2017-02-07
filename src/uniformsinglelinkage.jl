"""
Uniform distribution over single linkage matching matricies
"""
immutable UniformSingleLinkage{G <: Integer} <: Distributions.DiscreteMatrixDistribution
    nrow::G # Number of rows
    ncol::G # Number of columns
    t::G # Number of Matches
end

#### Outer constructors
#UniformSingleLinkage{G <: Integer}(nrow::G, ncol::G, t::G) = UniformSingleLinkage{G}(nrow, ncol, t)
UniformSingleLinkage(nrow::Real, ncol::Real, t::Real) = UniformSingleLinkage(promote(nrow, ncol, t)...)
UniformSingleLinkage(d::Integer, t::Integer) = UniformSingleLinkage(d, d, t)

#### Parameters

Distributions.params(d::UniformSingleLinkage) = (d.nrow, d.ncol, d.t)
@inline partype{G<:Integer}(d::UniformSingleLinkage{G}) = G

#### Evaluation

function Distributions.pdf(d::UniformSingleLinkage, M::MatchMatrix)
    if d.t != length(M.rows) || d.nrow != M.nrow || d.ncol != M.ncol
        return 0.0
    elseif d.t == 0
        return 1.0
    else
        return (factorial(d.t) * binomial(M.nrow, d.t) * binomial(M.ncol, d.t)) ^ -1.
    end
end

function Distributions.logpdf(d::UniformSingleLinkage, M::MatchMatrix)
    if d.t != length(M.rows)
        return -Inf
    elseif d.t ==  0
        return 0.0
    elseif d.t == M.nrow && d.t == M.ncol
        return -sum(log(1:d.t))
    elseif d.t == M.nrow && d.t != M.ncol
        return -sum(log((M.ncol - d.t + 1):M.ncol))
    elseif d.t != M.nrow && d.t == M.ncol
        return -sum(log((M.nrow - d.t + 1):M.nrow))
    else
        return -sum(log((M.nrow - d.t + 1):M.nrow)) - sum(log((M.ncol - d.t + 1):M.ncol)) + sum(log(1:d.t))
    end
end

#### Sampling

function Distributions.rand(d::UniformSingleLinkage)
    rows = StatsBase.sample(1:d.nrow, d.t, replace = false)
    cols = StatsBase.sample(1:d.ncol, d.t, replace = false)
    return MatchMatrix(rows, cols, d.nrow, d.ncol)
end


#### Test things run, move to unit tests
d = UniformSingleLinkage(5, 5, 3)
Distributions.pdf(d, MatchMatrix([1], [1], 5, 5)) == 0
Distributions.pdf(d, MatchMatrix([1, 2, 3], [1, 2, 3], 5, 5)) == (factorial(3) * binomial(5, 3) * binomial(5, 3)) ^ -1.

log(Distributions.pdf(d, MatchMatrix([1, 2, 3], [1, 2, 3], 5, 5))) == Distributions.logpdf(d, MatchMatrix([1, 2, 3], [1, 2, 3], 5, 5))

Distributions.params(d)
Distributions.rand(d)
