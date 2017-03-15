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
    elseif n < 2
        return zero(n)
    end
    out = 0.0
    for ii = 2:n
        out += log(ii)
    end
    return out
end
