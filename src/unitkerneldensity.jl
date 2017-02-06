"""
Define type for storing information about the results of a kernel density estimate
"""
type UnitKernelDensity
    x::Array{Float64, 1}
    y::Array{Float64, 1}
    bw::Float64
    npoints::Int64
end

"""
Compute a KDE using a normal kernel in range 0 to 1
"""
function unitkde_slow(data::Array{Float64, 1}, npoints::Int64, bw::Float64)
    d = Distributions.Normal(0.0, bw)
    x = collect(0.0:(npoints - 1.0)^(-1.0):1.0)
    y = Array{Float64}(npoints)
    for (ii, xi) in enumerate(x)
        y[ii] = mean(Distributions.pdf(d, data .- xi))
    end
    UnitKernelDensity(x, y, bw, npoints)
end

"""
Method to operate on a set of points instead of evenly space points
"""
function unitkde_slow(data::Array{Float64, 1}, points::Array{Float64, 1}, bw::Float64)
    d = Distributions.Normal(0.0, bw)
    y = Array{Float64}(length(points))
    for (ii, xi) in enumerate(points)
        y[ii] = mean(Distributions.pdf(d, data .- xi))
    end
    UnitKernelDensity(points, y, bw, length(points))
end

"""
Compute the density at a point using a normal KDE using linear interpolation
"""
function unitkde_interpolate(x::Float64, d::UnitKernelDensity)
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
Compute a KDE using a normal kernel in range 0 to 1
"""
function unitkde_tilted(data::Array{Float64, 1}, npoints::Int64, bw::Float64, θ::Float64)
    d = Distributions.Normal(θ, bw)
    x = collect(0.0:(npoints - 1.0)^(-1.0):1.0)
    y = Array{Float64}(npoints)
    for (ii, xi) in enumerate(x)
        y[ii] = mean(Distributions.pdf(d, data .- xi))
    end
    UnitKernelDensity(x, y, bw, npoints)
end

"""
Method to operate on a set of points instead of evenly space points
"""
function unitkde_slow(data::Array{Float64, 1}, points::Array{Float64, 1}, bw::Float64, θ::Float64)
    d = Distributions.Normal(θ, bw)
    y = Array{Float64}(length(points))
    for (ii, xi) in enumerate(points)
        y[ii] = mean(Distributions.pdf(d, data .- xi))
    end
    UnitKernelDensity(points, y, bw, length(points))
end

#rfft()
#irfft()
#Distributions.cf()
