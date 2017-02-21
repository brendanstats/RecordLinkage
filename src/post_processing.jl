"""
Save single MatchMatrix or Array{MatchMatrix, 1} to file
"""
function write_matchmatrix{M <: MatchMatrix}(filename::String, C::M; sep::Char = '\t')
    f = open(filename, "w")
    write(f, "row", sep, "column", "\n")
    for (rr, cc) in zip(C.rows, C.cols)
        write(f, string(rr, sep, cc), "\n")
    end
    close(f)
end

function write_matchmatrix{M <: MatchMatrix}(filename::String, C::Array{M, 1}; sep::Char = '\t')
    f = open(filename, "w")
    write(f, "index", sep, "row", sep, "column", "\n")
    write(f, string(0, sep, C[1].nrow, sep, C[1].ncol), "\n")
    for (ii, c) in enumerate(C)
        for (rr, cc) in zip(c.rows, c.cols)
            write(f, string(ii, sep, rr, sep, cc), "\n")
        end
    end
    close(f)
end

"""
Save M and U probabilities to file
"""
function write_probs{T <: AbstractFloat}(filename::String, M::Array{T, 1}, U::Array{T, 1}; sep::Char = '\t')
    if size(M, 1) != size(U, 1)
        error("M and U probabilities should have the same length")
    end
    f = open(filename, "w")
    write(f, "index", sep, "M", sep, "U", "\n")
    for (ii, (m, u)) in enumerate(zip(M, U))
        write(f, string(ii, sep, m, sep, u, "\n"))
    end
    close(f)
end


function write_probs{T <: AbstractFloat}(filename::String, M::Array{T, 2}, U::Array{T, 2}; sep::Char = '\t')
    if size(M, 1) != size(U, 1)
        error("M and U probabilities should have the same length")
    end
    if size(M, 2) != size(U, 2)
        error("M and U probabilities should have the same dimension")
    end
    n, d = size(M)
    f = open(filename, "w")
    write(f, "index")
    for ii in 1:d
        write(f, string(sep, "M", ii))
    end
    for ii in 1:d
        write(f, string(sep, "U", ii))
    end
    write(f, "\n")
    ct = 0
    while ct < n
        ct += 1
        write(f, string(ct))
        for ii in 1:d
            write(f, string(sep, M[ct, ii]))
        end
        for ii in 1:d
            write(f, string(sep, U[ct, ii]))
        end
        write(f, "\n")
    end
    close(f)
end

"""
Read text file created using write_matchmatrix(), present results in either summarized or
complete form
"""
function read_matchmatrix(filename::String; sep::Char = '\t', T::Type = Int64, summarize::Bool = true, revert::Bool = false)
    f = open(filename, "r")
    raw, labels = readdlm(f, sep, T, header = true)
    close(f)
    
    labels = vec(labels)
    n, m = size(raw)
    if m == 3 #indicates multiple matricies were saved
        nrow = raw[1, 2]
        ncol = raw[1, 3]
        nout = Base.maximum(raw[:, 1])
        if summarize #return totals within matrix
            out = zeros(T, nrow, ncol)
            ii = 1 #first row contains dimensions so exclude
            while ii < n
                ii += 1
                out[raw[ii, 2], raw[ii, 3]] += 1
            end
            return out, nout
        elseif revert
            
            startindx = Array{Int64}(nout)
            endindx = Array{Int64}(nout)
            startindx[raw[ii, 1]] = ii
            endindx[raw[n, 1]] = n
            ii = 2 #first row contains dimensions so exclude          
            while ii < n
                ii += 1
                if raw[ii, 1] != raw[ii - 1, 1]
                    startindx[raw[ii, 1]] = ii
                    endinx[raw[ii - 1, 1]] = ii - 1
                end
            end
            out = Array{MatchMatrix{T}}(nout)
            for (ii, (ff, ll)) in enumerate(zip(startindx, endindx))
                out[ii] = MatchMatrix(raw[ff:ll, 2], raw[ff:ll, 3], nrow, ncol)
            end
            return out, nout
        else
            return labels, raw[2:n, :], nrow, ncol
        end
    elseif m == 2 #indicates single matrix was saved
        nrow = raw[1, 1]
        ncol = raw[1, 2]
        if summarize
            out = zeros(T, nrow, ncol)
            ii = 1 #first row contains dimensions so exclude
            while ii < n
                ii += 1
                out[raw[ii, 1], raw[ii, 2]] += 1
            end
            return out, nout
        elseif revert
            return MatchMatrix(raw[2:n, 1], raw[2:n, 2], nrow, ncol)
        else
            return labels, raw[2:n, :], nrow, ncol
        end
    end
end
