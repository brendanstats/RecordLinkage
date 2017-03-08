function unconstrained_markov_model{G <: Integer}(rows::Array{G,1}, cols::Array{G,1}, nrows::Array{G, 1}, ncols::Array{G,1})
    rowentries = zeros(nrows)
    colentries = zeros(ncols)
    totalRows = sum(nrows)
    totalCols = sum(ncols)
    logP = 0.0
    for (t, (ii, jj)) in enumerate(zip(rows, cols))
        logP += log(nrows[ii] - rowentries[ii]) + log(ncols[jj] - colentries[jj])
        logP -= log(totalRows - t + 1) + log(totalCols - t + 1)
        rowentries[ii] += 1
        colentries[jj] += 1
    end
    return exp(logP)
end

function constrained_markov_model{G <: Integer}(rows::Array{G,1}, cols::Array{G,1}, exrows::Array{G, 1}, excols::Array{G, 1}, nrows::Array{G, 1}, ncols::Array{G,1})
    rowentries = zeros(nrows)
    colentries = zeros(ncols)
    totalRows = sum(nrows)
    totalCols = sum(ncols)
    logP = 0.0
    for (t, (ii, jj)) in enumerate(zip(rows, cols))
        logP += log(nrows[ii] - rowentries[ii]) + log(ncols[jj] - colentries[jj])
        exlinks = 0
        for (rr, cc) in zip(exrows, excols)
            exlinks += (nrows[rr] - rowentries[rr]) * (ncols[cc] - colentries[cc])
        end
        logP -= log((totalRows - t + 1) * (totalCols - t + 1) - exlinks)
        rowentries[ii] += 1
        colentries[jj] += 1
    end
    return exp(logP)
end

function adjusted_constrained_markov_model{G <: Integer}(rows::Array{G,1}, cols::Array{G,1}, exrows::Array{G, 1}, excols::Array{G, 1}, nrows::Array{G, 1}, ncols::Array{G,1})

    #setup rows to exclude
    kr, kc = length(nrows), length(ncols)
    inclmat = trues(kr, kc)
    for (ii, jj) in zip(exrows, excols)
        inclmat[ii, jj] = false
    end

    rowentries = zeros(nrows)
    colentries = zeros(ncols)
    totalRows = sum(nrows)
    totalCols = sum(ncols)
    logP = 0.0
    for (t, (ii, jj)) in enumerate(zip(rows, cols))
        logP += log(nrows[ii] - rowentries[ii]) + log(ncols[jj] - colentries[jj])
        exlinks = 0
        for (rr, cc) in zip(exrows, excols)
            exlinks += (nrows[rr] - rowentries[rr]) * (ncols[cc] - colentries[cc])
        end
        logP -= log((totalRows - t + 1) * (totalCols - t + 1) - exlinks)
        rowentries[ii] += 1
        colentries[jj] += 1
    end
    return exp(logP)
end

#Single entries
unconstrained_markov_model([1], [2], [3,2], [2,3])
unconstrained_markov_model([2], [1], [3,2], [2,3])

#Should match unconstrained since no constraints listed
constrained_markov_model([1], [2], Array{Int64}(0), Array{Int64}(0), [3,2], [2,3])
constrained_markov_model([2], [1], Array{Int64}(0), Array{Int64}(0), [3,2], [2,3])

#Gives normalized probabilities
constrained_markov_model([1], [2], [1, 2], [1, 2], [3,2], [2,3])
constrained_markov_model([2], [1], [1, 2], [1, 2], [3,2], [2,3])

#Ratios match in single case
unconstrained_markov_model([1], [2], [3,2], [2,3]) / (unconstrained_markov_model([1], [2], [3,2], [2,3]) + unconstrained_markov_model([2], [1], [3,2], [2,3]))
unconstrained_markov_model([2], [1], [3,2], [2,3]) / (unconstrained_markov_model([1], [2], [3,2], [2,3]) + unconstrained_markov_model([2], [1], [3,2], [2,3]))

#Two entries
unconstrained_markov_model([1, 1], [2, 2], [3,2], [2,3])
unconstrained_markov_model([1, 2], [2, 1], [3,2], [2,3])
unconstrained_markov_model([2, 1], [1, 2], [3,2], [2,3])
unconstrained_markov_model([2, 2], [1, 1], [3,2], [2,3])

constrained_markov_model([1, 1], [2, 2], Array{Int64}(0), Array{Int64}(0), [3,2], [2,3])
constrained_markov_model([1, 2], [2, 1], Array{Int64}(0), Array{Int64}(0), [3,2], [2,3])
constrained_markov_model([2, 1], [1, 2], Array{Int64}(0), Array{Int64}(0), [3,2], [2,3])
constrained_markov_model([2, 2], [1, 1], Array{Int64}(0), Array{Int64}(0), [3,2], [2,3])

constrained_markov_model([1, 1], [2, 2], [1, 2], [1, 2], [3,2], [2,3])
constrained_markov_model([1, 2], [2, 1], [1, 2], [1, 2], [3,2], [2,3])
constrained_markov_model([2, 1], [1, 2], [1, 2], [1, 2], [3,2], [2,3]) #matches first two in unconstrained case
constrained_markov_model([2, 2], [1, 1], [1, 2], [1, 2], [3,2], [2,3])


#Three entries
unconstrained_markov_model([2, 2, 2], [1, 1, 1], [3,2], [2,3])
