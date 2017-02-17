"""
Define a mixture between a UnitKernelDensity a parameterized distribution
"""
type UnitKDEMixture{D <: Distributions.ContinuousUnivariateDistribution, T<: AbstractFloat}
    ukde::UnitKernelDensity{T}
    d::D
    p::T
end
    
function Distributions.rand{M <: UnitKDEMixture}(ukdem::M)
    if rand() < ukdem.p
        return rand(ukdem.d)
    else
        return rand(ukdem.ukde)
    end
end

function Distributions.rand{M <: UnitKDEMixture}(ukdem::M, n::Int64)
    return [rand(ukdem) for ii in 1:n]
end

function Distributions.pdf{M <: UnitKDEMixture}(ukdem::M, x::Float64)
    return ukdem.p * Distributions.pdf(ukdem.d, x) + (1.0 - ukdem.p) * Distributions.pdf(ukdem.d, x)
end

function Distributions.pdf{M <: UnitKDEMixture}(ukdem::M, x::Array{Float64, 1})
    out = zeros(x)
    for (ii, xi) in enumerate(x)
        out[ii] = Distributions.pdf(ukdem, xi)
    end
    return out
end

function Distributions.logpdf{M <: UnitKDEMixture}(ukdem::M, x::Float64)
    return log(Distributions.pdf(ukdem, x))
end

function Distributions.logpdf{M <: UnitKDEMixture}(ukdem::M, x::Array{Float64, 1})
    return log(Distributions.pdf(ukdem, x))
end

"""
Beta distribution parameterized by mode and concentration
`beta_mode(ω, κ)`
Where ``0 ≤ ω ≤ 1`` and ``κ > 2``
"""
function beta_mode{T <: AbstractFloat}(ω::T, κ::T)
    α = ω * (κ - 2) + 1
    β = (1 - ω) * (κ - 2) + 1
    return Distributions.Beta(α, β)
end
