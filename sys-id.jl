import Pkg
Pkg.add("Plots")
Pkg.add("Distributions")
Pkg.add("RandomMatrices")

using Plots
using Random, LinearAlgebra
default(size=(500,280))
using Distributions
using RandomMatrices

mutable struct LDS
    inputDim::Int
    stateDim::Int
    outputDim::Int
    x::Array{Float64,1}
    stateCovar::Array{Float64,2}
    outputCovar::Array{Float64,2}
    A::Array{Float64,2}
    B::Array{Float64,2}
    C::Array{Float64,2}
    D::Array{Float64,2}
end

function step(lds::LDS, u::Array{Float64,1})
    #xi = if (lds.stateCovar == zeros(lds.stateDim,lds.stateDim)) zeros(lds.stateDim) else 
    #    rand(MvNormal(zeros(lds.stateDim),lds.stateCovar),1) end
    local xi
    local eta
    try
        xi = rand(MvNormal(zeros(lds.stateDim),lds.stateCovar),1)
    catch y
        xi = zeros(lds.stateDim)
    end
    #eta = if (lds.outputCovar == zeros(lds.outputDim,lds.outputDim)) zeros(lds.outputDim) else
    #    rand(MvNormal(zeros(lds.outputDim),lds.outputCovar),1) end
    try
        eta = rand(MvNormal(zeros(lds.outputDim),lds.outputCovar),1)
    catch y
        eta = zeros(lds.outputDim)
    end
    #display((lds.A,lds.x,lds.B,u,xi))
    lds.x = lds.A*lds.x + lds.B*u + xi
    lds.C*lds.x + lds.D*u + eta 
end

function ksvd(H,k)
    svdH = svd(H)
    H1 = svdH.U * Diagonal(svdH.S) * Diagonal([(if (i<=k) 1.0 else 0 end) for i in 1:size(H)[2]]) * svdH.Vt
    H1
end

function fmat(f,m,n)
    [f(i,j) for i in 1:m, j in 1:n]
end

function toep(a,b,F)
    if length(size(F))==2
        F=reshape(F,size(F)[1],1,size(F)[2])
    elseif length(size(F))==1
        F =reshape(F,size(F)[1],1,1)
    end
    #display(F)
    # starting with index 1 on diagonals
    sz = size(F)[2:end]
    #because hvcat does h first
    tiledMatrix = fmat((i,j)-> if (i<=j) F[j-i+1,:,:] else zeros(sz...) end, b, a)
    #tiledMatrix = fmat((i,j)-> if (i>=j) F[i-j+1,:,:] else zeros(sz...) end, a, b)
    #display(tiledMatrix)
    flatTiles = reshape(tiledMatrix, a*b)
    #display(flatTiles)
    TF = hvcat(b,flatTiles...)
    TF
end

function least_squares_F(l, us,Y)
    t = size(us)[1]
    U = toep(t,l+1, us)
    #display(U)
    #display(U'*U)
    #display(Y)
    MF = (U'*U) \ U'*Y
    #F = reshape(MF, )
    dy = size(Y)[2]
    du = size(us)[2]
    F = fill(0.0,l+1,dy,du)
    #display(MF)
    for i=1:l+1
        F[i,:,:]=MF[(i-1)*du+1:i*du,:]'
    end
    F
end

function collect_data(lds, T)
    us = fill(0.0,T,lds.inputDim)
    #println("us:")
    #println(us)
    Y = fill(0.0,T,lds.outputDim)
    for t=1:T
        u=rand(MvNormal(zeros(lds.inputDim),Matrix(I,lds.inputDim,lds.inputDim)))
        us[t,:]=u
        Y[t,:] = step(lds,u)
    end
    #println(us)
    return (us,Y)
end     

function hankel(m,n,F,offset=0)
    sz = size(F)
    rows = sz[2]
    cols = sz[3]
    tiledMatrix = fmat((i,j)-> F[i+j-1+offset,:,:], n,m)
    flatTiles = reshape(tiledMatrix, m*n)
    H = hvcat(n,flatTiles...)
    H
    #[f[i+j-1] for i in 1:m, j in 1:n]
end

function diag_avgs(m,n,rows,cols,A)
    cat([reshape(mean([A[(i-1)*rows+1:i*rows,(k-i-1)*cols+1:(k-i)*cols] for i in max(1,k-n):min(m,k-1)]),(1,rows,cols)) for k in 2:m+n]...,dims=1)
end

#will extract up to 3^levels
#F should be defined up to 4/3*3^levels-1
function hankel_multisvd(levels,r,F)
    Ftilde = F[1:3^levels,:,:]
    sz = size(F)[2:end]
    local Fl
    for l=1:levels
        s=2*3^(l-1)
        H=hankel(s,s,F)
        Hr=ksvd(H,r)
        #display(Hr)
        Fl=diag_avgs(s,s,sz[1],sz[2],Hr)
        Ftilde[3^(l-1)+1:3^l,:,:]=Fl[3^(l-1)+1:3^l,:,:]
    end
    Ftilde2 = Fl[1:3^levels,:,:] # 1 scale only
    Ftilde, Ftilde2
end

function fir(lds,l)
    #display([reshape(lds.C*lds.A^(t-1)*lds.B,(1,lds.outputDim,lds.inputDim)) for t=1:l])
    cat([reshape(lds.C*lds.A^(t-1)*lds.B,(1,lds.outputDim,lds.inputDim)) for t=1:l]...,dims=1)
end

function random_orth(d1,d2)
    A = rand(Haar(1),max(d1,d2))
    A[1:d1,1:d2]
end

function random_lds(d_u,d,d_y,maxEV)
    B = random_orth(d,d_u)
    C = random_orth(d_y,d)
    A = rand(MvNormal(zeros(d),Matrix(I,d,d)),d)
    vals, vecs = eigen(A,sortby=(x->-abs(x)))
    A = real.(vecs*Diagonal(vals/abs(vals[1])*maxEV) * inv(vecs))
    LDS(
        d_u, #inputDim
        d, #stateDim
        d_y, #outputDim
        zeros(d), #x
        zeros(d,d), #stateCovar
        Matrix{Float64}(I,d_y,d_y), #outputCovar
        A, #A
        B, #B
        C, #C
        zeros(d_y,d_u)) #D
end

function run_experiment_on(lds, T, lossEvery, levels, d;verbose=false,startAt=0)
    len = 4*3^(levels-1)-1
    display(len)
    (us,Y) = collect_data(lds, T)
    Ftrue = fir(lds,3^levels)
    numPts = Integer((T-startAt)/lossEvery)
    errF = fill(0.0,numPts)
    errSVD = fill(0.0,numPts)
    errMultiSVD = fill(0.0,numPts)
    local F
    local Ftilde2
    local Ftilde
    for i=1:Integer((T-startAt)/lossEvery)
        us1 = us[1:startAt+i*lossEvery,:]
        Y1 = Y[1:startAt+i*lossEvery,:]
        F = least_squares_F(len,us1,Y1)
        #display((size(F[1:3^levels,:,:]),size(Ftrue)))
        #display(F[1:3^levels,:,:]-Ftrue)
        #blah = F[1:3^levels,:,:]-Ftrue
        #display(norm(blah))
        #print(i)
        errF[i] = norm(F[1:3^levels,:,:]-Ftrue)
        #print(errF[i])
        #display(("F:",F,size(F)))
        #display(("Ftrue:",Ftrue))
        Ftilde, Ftilde2 = hankel_multisvd(levels,d,F)
        errSVD[i] = norm(Ftilde2-Ftrue)
        errMultiSVD[i] = norm(Ftilde-Ftrue)
        display((startAt+i*lossEvery, errF[i], errSVD[i], errMultiSVD[i]))
    end
    if verbose
        display("Ftrue:")
        display(Ftrue)
        display(F[1:3^levels,:,:])
        display(Ftilde2)
        display(Ftilde)
    end
    (F[1:3^levels,:,:], Ftilde2, Ftilde, errF, errSVD, errMultiSVD)
end

function run_experiment(T, lossEvery, levels, d_u, d, d_y, maxEV;verbose=false,startAt=0)
    run_experiment_on(random_lds(d_u,d,d_y,maxEV),T, lossEvery, levels, d;verbose=false,startAt=startAt)
end

function graph_experiment(lossEvery,errFs, errSVDs, errMultiSVDs;title="Error",startAt=0)
    pts = size(errFs)[2]
    errFavg = mean(errFs,dims=1)
    errSVDavg = mean(errSVDs,dims=1)
    errMultiSVDavg = mean(errMultiSVDs,dims=1)
    xs = startAt.+lossEvery*(1:pts)
    #display(xs)
    #display(errFavg)
    #display(cat(errFavg,errSVDavg,errMultiSVDavg,dims=1))
    plot(xs,cat(errFavg,errSVDavg,errMultiSVDavg,dims=1)', label = ["least squares" "SVD" "multiscale SVD"], lw=2, xlabel="time", ylabel="error", title=title)
end

function average_over_trials(d_u,d,d_y,levels,maxEV,trials,T,lossEvery;startAt=0)
    pts = Integer((T-startAt)/lossEvery)
    errFs = fill(0.0,trials,pts)
    errSVDs = fill(0.0,trials,pts)
    errMultiSVDs = fill(0.0,trials,pts)
    for trial=1:trials
        display(trial)
        (F, Ftilde2, Ftilde, errF, errSVD, errMultiSVD) = run_experiment(T, lossEvery, levels, d_u,d,d_y,maxEV, verbose=false,startAt=startAt)
        errFs[trial,:]=errF
        errSVDs[trial,:]=errSVD
        errMultiSVDs[trial,:]=errMultiSVD
    end
    (lossEvery,errFs, errSVDs, errMultiSVDs)
end

Random.seed!(1)
#average_over_trials(d_u,d,d_y,levels,maxEV,trials,T,lossEvery;startAt=0)
(lossEvery,errFs, errSVDs, errMultiSVDs)=average_over_trials(1,1,1,3,0.9,10,2000,100;startAt=0)

graph_experiment(lossEvery,errFs, errSVDs, errMultiSVDs, title="d=1, L=27")

savefig("lds-1-27.png")

Random.seed!(2)
(lossEvery,errFs, errSVDs, errMultiSVDs)=average_over_trials(3,3,3,3,0.9,10,2000,100,startAt=100)

graph_experiment(lossEvery,errFs, errSVDs, errMultiSVDs, title="d=3, L=27",startAt=100)

savefig("lds-3-27.png")

Random.seed!(3)
(lossEvery,errFs, errSVDs, errMultiSVDs)=average_over_trials(3,3,3,4,0.95,10,2000,100,startAt=300)

graph_experiment(lossEvery,errFs, errSVDs, errMultiSVDs, title="d=3, L=81",startAt=300)

savefig("lds-3-81.png")

Random.seed!(4)
(lossEvery,errFs, errSVDs, errMultiSVDs)=average_over_trials(3,5,3,4,0.95,10,2000,100,startAt=300)

graph_experiment(lossEvery,errFs, errSVDs, errMultiSVDs, title="d=5, L=81",startAt=300)

savefig("lds-5-81.png")

Random.seed!(5)
(lossEvery,errFs, errSVDs, errMultiSVDs)=average_over_trials(3,10,3,4,0.95,10,2000,100,startAt=300)

graph_experiment(lossEvery,errFs, errSVDs, errMultiSVDs, title="d=10, L=81",startAt=300)

savefig("lds-10-81.png")