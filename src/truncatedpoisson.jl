immutable TruncatedPoisson{T <: Real, G <: Integer} <: Distributions.DiscreteUnivariateDistribution
    λ::T
    tmax::G
    α::T
end

#### Outer constructors
TruncatedPoisson(λ::Integer, tmax::Integer, α::Integer) = TruncatedPoisson(Float64(λ), tmax, Float64(α))

function TruncatedPoisson{T <: Real, G <: Integer}(λ::T, tmax::G)
    tot = 1.0
    term = 1.0
    for ii in 1:tmax
        term *= λ / ii
        tot += term
    end
    return TruncatedPoisson{T, G}(λ, tmax, exp(λ - log(tot)))
end

@Distributions.distr_support TruncatedPoisson 0 (d.λ == zero(typeof(d.λ)) ? 0 : d.tmax)

#### Parameters

Distributions.params(d::TruncatedPoisson) = (d.λ, d.tmax, d.α)
@inline partype{T <: Real, G <: Integer}(d::TruncatedPoisson{T, G}) = (T, G)

#### Statistics

#### Evaluation
Distributions.pdf(d::TruncatedPoisson, x::Int64) = d.α * exp(x * log(d.λ) - d.λ - sum(log(1:x)))
Distributions.pdf(d::TruncatedPoisson, x::Int32) = d.α * exp(x * log(d.λ) - d.λ - sum(log(1:x)))
Distributions.logpdf(d::TruncatedPoisson, x::Int64) = log(d.α) - d.λ + x * log(d.λ) - sum(log(1:x))
Distributions.logpdf(d::TruncatedPoisson, x::Int32) = log(d.α) - d.λ + x * log(d.λ) - sum(log(1:x))
#### Sampling

function Distributions.rand(d::TruncatedPoisson)
    p = rand()
    tot = 0.
    for kk in 0:d.tmax
        tot += Distributions.pdf(d, kk)
        if tot > p
            return kk
        end
    end
end

#### Test things run, move to unit tests
d = TruncatedPoisson(1.0, 1)
d.tmax == 1
d.λ == 1
d.α == 1. / (1.0^1*exp(-1.0) / factorial(0) + 1.0^1*exp(-1.0) / factorial(1))
Distributions.params(d)
Distributions.pdf(d, 0)
Distributions.logpdf(d, 0)

Distributions.rand(d)
Distributions.rand(TruncatedPoisson(2.0, 20))
Distributions.pdf(TruncatedPoisson(2.0, 20), 0)
Distributions.pdf(TruncatedPoisson(2.0, 20), 1)
Distributions.pdf(TruncatedPoisson(2.0, 20), 2)
Distributions.pdf(TruncatedPoisson(2.0, 20), 3)
TruncatedPoisson(200.0, 1000)

Distributions.rand(TruncatedPoisson(300.0, 1000))
