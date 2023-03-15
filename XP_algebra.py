import numpy as np
from NSpace import *
from common import * 

def XP2Str(a,N=2):
    p,x,z = XPComponents(a)
    x,z = ZMat2str(x),ZMat2str(z)
    return f'XP_{N}({p}|{x}|{z})'

def XPList2Str(G,N=2):
    return "\n".join([XP2Str(a,N) for a in G])

def XPn(a):
    return len(a) // 2 - 1

def XPComponents(a):
    n = XPn(a)
    x,z = a[:n+1],a[n+1:]
    p = x[-1] + z[-1] * 2
    return p,x[:-1],z[:-1]

def XPfromComponents(p,x,z):
    px = p % 2
    pz = p // 2
    return np.hstack([x,[px],z,[pz]])

def XPRound(a,N=2):
    n = XPn(a)
    x,z = np.mod(a[:n+1],2),np.mod(a[n+1:],N)
    return np.hstack([x,z])

def XPMul(a,b,N=2):
    c = a + b
    n = XPn(a)
    p1,x1,z1 = XPComponents(a)
    p2,x2,z2 = XPComponents(b)
    c += XPD(z1*x2,N)
    ## adjust phase component
    c[-1] += a[n] * b[n]
    return XPRound(c,N)

def XPD(z,N=2):
    n = len(z)
    return XPRound(np.hstack([ZMatZeros(n+1),-2*z,[np.sum(z)]]),N)

def XPComm(a,b,N=2):
    p1,x1,z1 = XPComponents(a)
    p2,x2,z2 = XPComponents(b)
    return XPD(x1*z2 - x2*z1 + 2*x1*x2*(z1-z2),N)   

def genMatrix(SX,SZ,SXZ,randomsign=True):
    '''Make generator matrix from codetables.de format data'''
    r,n = np.shape(SX)
    ## sign adjustments for X stabilisers to ensure they square to identity
    s = [np.sum(SX[i]*SXZ[i]) % 2 for i in range(r)]
    G = np.hstack([SX,np.transpose([s]),SXZ,ZMatZeros((r,1))])
    ## Z stabilisers
    s,n = np.shape(SZ)
    if s > 0:
        SZ = np.hstack([ZMatZeros((s,n+1)),SZ,ZMatZeros((s,1))])
        G = np.vstack([G,SZ])
    if randomsign:
        ## allocate signs of -1 randomly to stabilisers
        s = np.random.randint(2,size=len(G))
        G[:,-1] = s
    return G

def XPSimplifyX(G,N=2):
    n = XPn(G[0])
    A = G[:,:n+1]
    Sx,rowops = How(A,2)
    U,K = GetUK(A,Sx,rowops,2)
    U = np.vstack([U,K])
    G = [XPGenProd(G,u,N) for u in U]
    return ZMat(G)

def XPisDiag(a):
    n = XPn(a)
    return 1 if np.sum(a[:n]) == 0 else 0


def XPGenProd(A,u,N=2):
    '''A is a list of XP operators. 
    u is a binary vector indicating power u[i] of the operator A[i]'''
    m,n = np.shape(A)
    ## starting value = I
    temp = ZMatZeros(n)
    for i in bin2Set(u):
        temp = XPMul(temp,A[i],N)
    return temp


def canonicalGenerators(G):
    '''Return canonical form stabiliser generators and logical X/Z following Neilsen & Chuang page 471
    G is the generator matrix in binary form.'''
    N=2
    n = XPn(G[0])

    ## X components in RREF
    G = XPSimplifyX(G,N)

    ## Split into diagonal and non-diagonal operators
    ix = ZMat([XPisDiag(a) for a in G])
    SX,SZ = G[bin2Set(1-ix)],G[bin2Set(ix)]
    ## Split into X and Z components
    Sx = SX[:,:n+1]
    Sxz = SX[:,n+1:]
    Sz = SZ[:,n+1:]

    ## Move leading indices of Sx to right
    LiX = [leadingIndex(a) for a in Sx]
    ix = sorted(set(range(n)) - set(LiX)) + LiX + [n]

    ## reorder columns
    Sx = Sx[:,ix]
    Sxz = Sxz[:,ix] 
    Sz = Sz[:,ix] 

    ## Sz in Howell form/RREF
    Sz, rowops = How(Sz,N)
    ## eliminate entries in Sxz
    for i in range(len(Sxz)):
        Sxz[i],u = matResidual(Sz,Sxz[i],N)

    ## move leading indices of Sz to right
    LiZ = [leadingIndex(a) for a in Sz]
    ## remember the original positions of LiZ
    LiZR = [ix[i] for i in LiZ]
    ix = sorted(set(range(n)) - set(LiZ)) + LiZ + [n]
    ## reorder columns
    Sx = Sx[:,ix]
    Sxz = Sxz[:,ix] 
    Sz = Sz[:,ix] 

    ## find LX, LZ using method in Neilsen & Chuang, page 471
    r = len(Sx)
    s = len(Sz)
    k = n - r - s 
    A2 = Sx[:,:k]   
    C = Sxz[:,:k]
    E = Sz[:,:k]
    Ik = ZMatI(k)
    pVec = ZMatZeros((k,1))
    Lx = np.hstack([Ik,ZMatZeros((k,r)),E.T,pVec])
    Lxz = np.hstack([ZMatZeros((k,k)),C.T,ZMatZeros((k,s)),pVec])
    Lz = np.hstack([Ik,A2.T,ZMatZeros((k,s)),pVec])
    
    ## reorder columns and turn into operators
    ix = sorted(set(range(n)) - set(LiX) - set(LiZR))  + LiX + LiZR + [n]
    ixR = ixRev(ix)
    SX = np.hstack([Sx[:,ixR],Sxz[:,ixR]])
    SZ = np.hstack([ZMatZeros((s,n+1)),Sz[:,ixR]])
    LX = np.hstack([Lx[:,ixR],Lxz[:,ixR]])
    LZ = np.hstack([ZMatZeros((k,n+1)),Lz[:,ixR]])
    return SX,SZ,LX,LZ

def checkCommRelations(SX,SZ,LX,LZ):
    '''For stabiliser generators and logical X/Z, check that the correct commutation relations are met.'''
    G = np.vstack([SX,SZ,LX,LZ])
    gLen = len(G)

    ## C[i,j] = 0 iff G[i] commutes with G[j] 
    C = ZMatZeros((gLen,gLen))
    for i in range(gLen):
        for j in range(gLen):
            c = XPComm(G[i],G[j],2)
            ## commutator sign
            C[i,j] = c[-1]
    # print('Commutator relations SX,SZ,LX,LZ')
    # print(ZmatPrint(C))

    r,s,k = len(SX),len(SZ),len(LX)
    for i in range(k):
        ## check that LX[i] and LZ[i] anticommute
        if C[r+s+i,r+s+i+k] != 1:
            return False
        if C[r+s+i+k,r+s+i] != 1:
            return False   
        C[r+s+i,r+s+i+k] = 0     
        C[r+s+i+k,r+s+i] = 0
    ## check that everything else commutes
    return np.sum(C) == 0