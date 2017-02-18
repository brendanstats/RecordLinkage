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
