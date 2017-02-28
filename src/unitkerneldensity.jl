"""
Define type for storing information about the results of a kernel density estimate
"""
type UnitKernelDensity{T <: AbstractFloat}
    x::Array{T, 1}
    y::Array{T, 1}
    cdf::Array{T, 1}
    bw::T
    npoints::Int64
end

"""
Compute a KDE using a normal kernel in range 0 to 1
"""
function unitkde_slow(data::Array{Float64, 1}, npoints::Int64, bw::Float64)
    d = Distributions.Normal(0.0, bw)
    x = collect(linspace(0.0, 1.0, npoints))
    stepsize = x[2] - x[1]
    y = Array{Float64}(npoints)
    #Compute point densities
    for (ii, xi) in enumerate(x)
        y[ii] = mean(Distributions.pdf(d, data .- xi))
    end
    #Compute approximate CDF
    approxcdf = zeros(y)
    approxcdf[1] = 0.0
    for ii in 2:npoints
        approxcdf[ii] = approxcdf[ii - 1] + 0.5 * stepsize * (y[ii] + y[ii - 1])
    end
    N = approxcdf[end]^(-1.0)
    return UnitKernelDensity(x, N .* y, N .* approxcdf, bw, npoints)
end

"""
Method to operate on a set of points instead of evenly space points
"""
function unitkde_slow(data::Array{Float64, 1}, points::Array{Float64, 1}, bw::Float64)
    d = Distributions.Normal(0.0, bw)
    y = zeros(points)
    #Compute point densities
    for (ii, xi) in enumerate(points)
        y[ii] = mean(Distributions.pdf(d, data .- xi))
    end
    approxcdf = zeros(points)
    #Compute approximate CDF
    for ii in 2:length(points)
        approxcdf[ii] = approxcdf[ii - 1] + 0.5 * (points[ii] - points[ii - 1]) * (y[ii] + y[ii - 1])
    end
    N = approxcdf[end]^(-1.0)
    return UnitKernelDensity(points, N .* y, N .* approxcdf, bw, length(points))
end

"""
Compute a KDE using a normal kernel in range 0 to 1
"""
function unitkde_tilted(data::Array{Float64, 1}, npoints::Int64, bw::Float64, θ::Float64)
    d = Distributions.Normal(θ, bw)
    x = collect(linspace(0.0, 1.0, npoints))
    stepsize = x[2] - x[1]
    y = Array{Float64}(npoints)
    #Compute point densities
    for (ii, xi) in enumerate(x)
        y[ii] = mean(Distributions.pdf(d, data .- xi))
    end
    #Compute approximate CDF
    approxcdf = zeros(y)
    approxcdf[1] = 0.0
    for ii in 2:npoints
        approxcdf[ii] = approxcdf[ii - 1] + 0.5 * stepsize * (y[ii] + y[ii - 1])
    end
    N = approxcdf[end]^(-1.0)
    return UnitKernelDensity(x, N .* y, N .* approxcdf, bw, npoints)
end

"""
Method to operate on a set of points instead of evenly space points
"""
function unitkde_tilted(data::Array{Float64, 1}, points::Array{Float64, 1}, bw::Float64, θ::Float64)
    d = Distributions.Normal(θ, bw)
    y = zeros(points)
    #Compute point densities
    for (ii, xi) in enumerate(points)
        y[ii] = mean(Distributions.pdf(d, data .- xi))
    end
    approxcdf = zeros(points)
    #Compute approximate CDF
    for ii in 2:length(points)
        approxcdf[ii] = approxcdf[ii - 1] + 0.5 * (points[ii] - points[ii - 1]) * (y[ii] + y[ii - 1])
    end
    N = approxcdf[end]^(-1.0)
    return UnitKernelDensity(points, N .* y, N .* approxcdf, bw, length(points))
end

"""
Sample from Linearly Interpolated UnitKernelDensity
"""
function Distributions.rand(d::UnitKernelDensity)
    r = rand()
    k = searchsortedfirst(d.cdf, r)
    return (r - d.cdf[k - 1]) / (d.cdf[k] - d.cdf[k - 1]) * (d.x[k] - d.x[k - 1]) + d.x[k - 1]
end

function Distributions.rand(d::UnitKernelDensity, n::Integer)
    out = Array{Float64}(n)
    for ii in 1:n
        out[ii] = Distributions.rand(d)
    end
    return out
end

"""
Compute the density at a point using a normal KDE using linear interpolation
"""
function Distributions.pdf(d::UnitKernelDensity, x::Float64)
    if x < 0.0 || 1.0 < x
        error("x must be between 0.0 and 1.0")
    end
    k = searchsortedfirst(d.x, x)
    if k == 1
        return d.y[1]
    end
    return (x - d.x[k - 1]) / (d.x[k] - d.x[k - 1]) * (d.y[k] - d.y[k - 1]) + d.y[k - 1]
end

"""
Find mode
"""
function Distributions.mode(d::UnitKernelDensity)
    return d.x[indmax(d.y)]
end

#rfft()
#irfft()
#Distributions.cf()
