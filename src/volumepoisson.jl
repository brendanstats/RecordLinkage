immutable VolumePoisson{T <: Real, G <: Integer} <: Distributions.DiscreteUnivariateDistribution
    θ::T
    n::G
    m::G
    α::Float64
end

#### Outer constructors
function VolumePoisson{T <: Real, G <: Integer}(θ::T, n::G)
    inc = n * θ + sum(log(1:n))
    tot = exp(-inc)
    for ii in 1:n
        inc += -θ - log(n + 1 - ii) + 2 * log(ii)
        tot += exp(-inc)
    end
    return VolumePoisson{T, G}(θ, n, n, 1. / tot)
end

function VolumePoisson{T <: Real, G <: Integer}(θ::T, n::G, m::G)
    if n == m
        return VolumePoisson(θ, n)
    else
        lower = min(n,m)
        upper = max(n, m)
        diff = upper - lower
        inc = n * θ + sum(log(1:n)) + sum(log(1:diff))
        tot = exp(-inc)
        for ii in 1:lower
            inc += -θ - log(n + 1 - ii) + log(ii) + log(diff + ii)
            tot += exp(-inc)
        end
        return VolumePoisson{T, G}(θ, n, n, 1. / tot)
    end
end

function coefficients(d::VolumePoisson)
    maxL = min(d.n,d.m) - 1
    out = Array{Float64}(maxL + 1)
    for ii in 0:maxL
        out[ii+1] = (d.n - ii) * (d.m - ii) / (ii + 1) * exp(-d.θ)
    end
    return out
end

@Distributions.distr_support VolumePoisson 0 (d.θ == zero(typeof(d.θ)) ? 0 : min(d.n, d.m))

#### Parameters

Distributions.params(d::VolumePoisson) = (d.θ, d.n, d.m, d.α)
@inline partype{T <: Real, G <: Integer}(d::VolumePoisson{T, G}) = (T, G)

#### Statistics

#### Evaluation
Distributions.pdf(d::VolumePoisson, x::Int64) = d.α * exp(-x * d.θ - sum(log(1:x)) - sum(log(1:(d.n - x))) - sum(log(1:(d.m - x))))
Distributions.pdf(d::VolumePoisson, x::Int32) = d.α * exp(-x * d.θ - sum(log(1:x)) - sum(log(1:(d.n - x))) - sum(log(1:(d.m - x))))
Distributions.logpdf(d::VolumePoisson, x::Int64) = log(d.α) - x * d.θ - sum(log(1:x)) - sum(log(1:(d.n - x))) - sum(log(1:(d.m - x)))
Distributions.logpdf(d::VolumePoisson, x::Int32) = log(d.α) - x * d.θ - sum(log(1:x)x) - sum(log(1:(d.n - x))) - sum(log(1:(d.m - x)))

#### Sampling

function Distributions.rand(d::VolumePoisson)
    p = rand()
    tot = 0.
    for kk in 0:min(d.n, d.m)
        tot += Distributions.pdf(d, kk)
        if tot > p
            return kk
        end
    end
end

#### Test things run, move to unit tests
d1 = VolumePoisson(2, 100, 100)
d2 = VolumePoisson(1.5, 1, 1)
d3 = VolumePoisson(2, 1000, 1000)
coefficients(d3)[1:10]
Distributions.pdf(d1, 1) / Distributions.pdf(d1, 0)


Distributions.rand(d1)
Distributions.rand(d2)

Distributions.pdf(d1, 70)
Distributions.pdf(d1, 75)
Distributions.pdf(d1, 80)

Distributions.pdf(d2, 0)
Distributions.pdf(d2, 1)
d2.α

1.0 / (1.0 / (factorial(0) * factorial(1) * factorial(1)) + exp(-1.5) / (factorial(1) * factorial(0) * factorial(0)))
d2.α

#=
for ii in 0:25:1000
    dens = -1.5 * ii - sum(log(1:ii)) - 2 * sum(log(1:(1000 - ii)))
    println(ii, ": ", dens)
end
=#
