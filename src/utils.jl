"""
Function to compute the sum of logs i.e. takes log(a), log(b) and returns log(a + b) in log space
"""
function logsum{G <: Real}(a::G, b::G; logged::Bool = true)
    if logged
        return a + log(1.0 + exp(b - a))
    else
        la = log(a)
        return la + log(1.0 + exp(log(b) - la))
    end
end

function logsum{G <: Real}(x::Array{G, 1}; logged::Bool = true)
    if logged
        logmax = Base.maximum(x)
        tot = 0.0
        for xi in x
            tot += exp(xi - logmax)
        end
        return logmax + log(tot)
    else
        logmax = log(Base.maximum(x))
        tot = 0.0
        for xi in x
            tot += exp(log(xi) - logmax)
        end
        return logmax + log(tot)
    end
end

"""
Return the log(n!)
"""
function logfactorial{G <: Integer}(n::G)
    if n < zero(n)
        error("n must be non-negative")
    elseif n < oftype(n, 2)
        return zero(n)
    end
    out = zero(Float64)
    for ii = oftype(n, 2):n
        out += log(ii)
    end
    return out
end

"""
Convert a 2D Array into a single column and add columns indicating row and column index in original array.  Order is row, col, value
"""
function grid2array{G <: Real}(x::Array{G, 2})
    nrow, ncol = size(x)
    return [repeat(1:nrow, outer=ncol) repeat(1:ncol, inner=nrow) vec(x)]
end


"""
Switch two indicies in a vector
"""
function switchidx!{G <: Integer, T <: Real}(v::Array{T, 1}, idx1::G, idx2::G)
    val = v[idx1]
    v[idx1] = v[idx2]
    v[idx2] = val
    return v
end

switchidx{G <: Integer, T <: Real}(v::Array{T, 1}, idx1::G, idx2::G) = switchidx!(copy(v), idx1, idx2)
