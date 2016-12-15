import StatsBase
import Distributions

immutable LogisticNormal{T <: Real} <: Distributions.ContinuousUnivariateDistribution
    x0::T
    σ::T
    #LogisticNormal(x0, σ) = (@Distributions.check_args(LogisticNormal, zero(x0) <= x0 && x0 <= zero(x0) && σ > zero(σ)); new(x0, σ))
end

#### Outer constructors
LogisticNormal{T <: Real}(x0::T, σ::T) = LogisticNormal{T}(x0, σ)
LogisticNormal(x0::Real, σ::Real) = LogisticNormal(promote(x0, σ)...)
LogisticNormal(x0::Integer, σ::Integer) = LogisticNormal(Float64(x0), Float64(σ))
LogisticNormal(x0::Real) = LogisticNormal(x0, 1.0)
LogisticNormal() = LogisticNormal(0.5, 1.0)

@Distributions.distr_support LogisticNormal 0.0 1.0

#### Parameters

params(d::LogisticNormal) = (d.x0, d.σ)
@inline partype{T<:Real}(d::LogisticNormal{T}) = T

#### Statistics

median(d::LogisticNormal) = d.x0

#### Evaluation
function logit(p::AbstractFloat)
    return log(p / (1.0 - p))
end

function logistic(α::AbstractFloat)
    return exp(α) / (exp(α) + 1.0)
end


Distributions.pdf(d::LogisticNormal, x::Real) = exp(-(logit(x) - logit(d.x0))^2 / (2 * d.σ^2)) / (sqrt(2 * pi) * d.σ) * (1/x + 1/(1 - x))

#### Sampling

Distributions.rand(d::LogisticNormal) = logistic(logit(d.x0) + d.σ * randn())

d = LogisticNormal(0.5, 1.0)
params(d)
median(d)
Distributions.rand(d)
Distributions.pdf(d, 0.45351992889361753)
