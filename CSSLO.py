import itertools as iter
from common import *
from NHow import *
from code_library import *
from distance import *
from XCP_algebra import *

##########################################
## CSS Codes
##########################################

def CSSCode(SX,LX=None,SZ=None,LZ=None,simplifyGens=False):
    """Create CSS code from various input types.

    Keyword arguments:
    SX -- X-checks 
    LX -- X-logicals (optional)
    SZ -- Z-checks (optional)
    LZ -- Z-logicals (optional)
    SimplifyGens -- return generators and logical Paulis in RREF/simplified form

    Default is to give values for SX and LX, then calculate SZ and LZ.
    Can take either text or arrays as input.
    Returns SX,LX,SZ,LZ. 
    """
    SX = bin2ZMat(SX)
    LX = bin2ZMat(LX)
    SZ = bin2ZMat(SZ)

    ## find n - number of qubits
    n = np.max([0 if A is None else A.shape[1] for A in [SX,LX,SZ]])
    SX = ZMatZeros((0,n)) if SX is None else ZMat(SX,n)

    ## Input is SX, SZ
    if LX is None:
        ## SZ could be None - handle this case
        SZ = ZMatZeros((0,n)) if SZ is None else ZMat(SZ,n)
        ## Calculate LX from SX, SZ
        LX = CSSgetLX(SX,SZ)
    ## Input is SX, LX
    elif SZ is None:
        LX = ZMatZeros((0,n)) if LX is None else ZMat(LX,n)
        ## Calculate SZ from SX, LX
        SZ = getK(np.vstack([SX,LX]),2)
    ## Find LZ
    if LZ is None:
        LZ = getK(SX,2)
        ## canonical form of LX - LX LZ^T = I_k
        LZ,lx = LXZDiag(LZ,LX)
    SZ = ZMat(SZ,n)
    LZ = ZMat(LZ,n)
    LX = ZMat(LX,n)
    if simplifyGens:
        SX = indepLZ(None,SX)
        LX = indepLZ(SX, LX)
        SZ = indepLZ(None,SZ)
        LZ = indepLZ(SZ, LZ)
    return SX,LX,SZ,LZ

def CSSgetLX(SX, SZ):
    """Get LX for CSS code with check matrices SX, SZ."""
    SXLX = getK(SZ,2)
    R, V = HowRes(SX,SXLX,2)
    LX = getH(R,2)
    return LX
	
def LXZCanonical(LX, LZ):
    """Modify LX, LZ such that LX LZ^T = I if possible."""
    LX, LZ = LXZDiag(LX,LZ)
    LZ, LX  = LXZDiag(LZ,LX)
    return LX, LZ

def LXZDiag(LX,LZ):
    """Convert LX to a form with minimal overlap with LZ. helper for LXZCanonical"""
    r, n = LZ.shape
    A = np.hstack([matMul(LX, LZ.T,2),LX])
    ## RREF of first r columns
    A = getH(A,2,r)
    ## updated LX is last r columns
    LX = A[:r,r:]
    return LX, LZ    

#####################################################
## Non-CSS codes
#####################################################

def XZ2LF(S):
    '''Return canonical binary linear form of a non-CSS code
    S: stabiliser generators 
    CE: binary linear code
    D: symmetric matrix rep S and CZ operators
    A: binary matrix rep CZ
    '''
    N=2
    tB = 2
    m,n = S.shape
    nB = n//tB
    ## Form RREF and find leading indices of X-components
    S,LiX = getH(S,N,nC=nB,retPivots=True)

    ## Move leading indices to left
    r = len(LiX)
    ix = np.hstack([LiX, invRange(nB,LiX)])
    S = ZMatPermuteCols(S,ix,tB=tB)

    ## Apply Hadamard to all qubits to right of LiX
    S = XZhad(S,np.arange(r,nB))

    ## Form RREF and find leading indices of Z-components
    S,LiZ = getH(S,N,retPivots=True)

    ## move leading indices of Sx and Sz to left
    s = len(LiZ) - len(LiX)
    k = nB - r - s
    ix2 = np.hstack([LiZ, invRange(nB,LiZ)])
    S = ZMatPermuteCols(S,ix2,tB=tB)

    ## update permutation
    ix = ix[ix2]
    ## Canonical matrices
    Sx, Sz = S[:,:nB],S[:,nB:]
    ## find Symmetric matrix D
    B = Sz[:r,:r]
    C = Sx[:r,nB-k:]
    A2 = Sz[:r,nB-k:]
    # D = np.mod(B + C @ A2.T, 2)
    D = matMul(C,A2.T,2 ) ^ B

    CE = Sx[:,nB-k:]
    A = Sz[:r,r:]
    return CE, A, D, ixRev(ix) 


def LF2XZ(CE,A,S,ix=None):
    '''Generate stabiliser code from Binary Linear form:
    CE: an (r+s) x s binary matrix representing a binary linear code
    A: an r x (s+k) binary matrix representing CZ operators
    S: an r x r symmetric binary matrix representing S and CZ operators
    ix: a permutation of the qubits'''
    tB = 2
    ## number of X-gens r, Z-gens s and logical qubits k
    r,sk = np.shape(A)
    rs,k = np.shape(CE)
    s = rs-r
    n = r + s + k
    ## form X-components Sx = (I | CE) - binary linear code
    Sx = np.hstack([ZMatI(rs),CE])
    ## form symmetric binary matrix representing applictaion of CZ and S operators
    D1 = np.hstack([S,A])
    D2 = np.hstack([A.T,ZMatZeros((sk,sk))])
    D = np.vstack([D1,D2])

    ## Generate Z-components Sz
    Sz = matMul(Sx, D, 2)

    ## Apply Had to last s + k qubits (swap X and Z components)
    S = np.hstack([Sx,Sz],dtype=np.int16)
    S = XZhad(S,np.arange(r,n,dtype=int))
    A2 = A[:,s:]
    Lz = np.hstack([ZMatZeros((k,n)),A2.T,ZMatZeros((k,s)),ZMatI(k)])
    E = CE[r:,:]
    C = CE[:r,:]
    Lx = np.hstack([ZMatZeros((k,r)),E.T,ZMatI(k),C.T,ZMatZeros((k,s+k))])
    ## Apply permutation of columns
    if ix is not None:
        return ZMatPermuteCols(S,ix,tB=tB),ZMatPermuteCols(Lx,ix,tB=tB),ZMatPermuteCols(Lz,ix,tB=tB)
    else:
        return S, Lx, Lz

def bin2XZ(n,k,r,x,Sdiag=True):
    '''Convert binary vector x to non-CSS [[n,k]] code with r independent X stabilisers'''
    Dbits = r*(r+1)//2 if Sdiag else r*(r-1)//2
    Abits = r*(n-r)
    Lbits = (n-k) * k
    d = Dbits+Abits+Lbits
    if typeName(x)[:3] == "int":
        x = int2bin(x,d)    
    x = np.array(x,dtype=np.int16)
    D = makeSymmetricMatrix(r,x[:Dbits],Sdiag)
    A = np.reshape(x[Dbits:Dbits+Abits],(r,n-r))
    L = np.reshape(x[-Lbits:],(n-k,k))
    return LF2XZ(L,A,D)

    
def XZhad(E, ix):
    '''Apply Hadamard to qubits according to index ix for Pauli operator list E'''
    m,n = E.shape
    nB = n//2
    ixN = ix + nB
    ix1 = np.arange(n)
    ix1[ix] = ixN
    ix1[ixN] = ix
    return ZMatPermuteCols(E,ix1)

def makeSymmetricMatrix(r,x,Sdiag=True):
    '''convert binary vector x to a symmetrix rxr matrix
    if Sdiag is False, the diagonal is all zeros'''
    S = ZMatZeros((r,r))
    if Sdiag:
        S[np.triu_indices(r, 0)] = x
    else:
        S[np.triu_indices(r, 1)] = x
    S[np.tril_indices(r, -1)] = S.T[np.tril_indices(r, -1)]
    return S

###################################################
## Supporting Methods for Logical Operator Algs  ##
###################################################

def Orbit2distIter(SX,t=None,return_u=False):
    '''Interator yielding binary rows of form (u SX mod 2) for wt(u) <= t.
    if return_u, yield u as well as the row.'''
    r, n = np.shape(SX)
    if t is None:
        t = r
    t = min(t, r)
    for k in range(t+1):
        for xSupp in iter.combinations(range(r),k):
            vSX = np.mod(np.sum(SX[xSupp,:],axis=0),2)
            if return_u:
                u = set2Bin(r,xSupp)
                yield vSX, u
            else:
                yield vSX

def Orbit2dist(SX,t=None,return_u=False):
    '''Matrix with binary rows of form (q + u SX mod 2) for wt(u) <= t.
    if return_u, yield u as well as the row.'''
    temp = list(Orbit2distIter(SX,t,return_u))
    if return_u:
        temp = list(zip(*temp))
        return [ZMat(a) for a in temp]
    else:
        return ZMat(temp)

##################################################
## Print stabilisers and codewords
##################################################

def print_SXLX(SX,LX,SZ,LZ,compact=True):
    '''Print the X-checks, X-logicals, Z-checks, Z-logicals of a CSS code.
    If compact=True, print full vector representations, otherwise print support of the vectors.'''
    k,n = LX.shape
    supercompact = (n > 70)
    opDict = {'SX':SX,'SZ':SZ,'LX':LX,'LZ':LZ}
    for AName,AList in opDict.items():
        print(AName)
        if supercompact:
            print(freqTable(np.sum(AList,axis=-1)))
        else:
            for x in AList:
                print(x2Str(x) if compact else ZMat2str(x))


def codewords(SX,LX):
    '''Return canonical codewords LI={v}, CW={sum_u (uSX + vLX)}'''
    r,n = np.shape(SX)
    OSX = Orbit2dist(SX)
    LI, CW = [],[]
    for m,v in Orbit2distIter(LX,return_u=True):
        S = np.mod(m + OSX,2)
        LI.append(v)
        CW.append(S)
    return LI, CW

def print_codewords(SX,LX):
    '''Print the canonical codwords of a CSS code defined by X-checks SX and X-logicals LX'''
    V, CW = codewords(SX,LX)
    print('\nCodewords')
    for i in range(len(V)):
        print(f'{ket(V[i])} : {state2str(CW[i])}')

def state2str(S):
    '''Print a state corresponding to a binary matrix S in the form sum_{x in S}|x>'''
    return "+".join([ket(x) for x in S])

def ket(x):
    '''Display |x> for states.'''
    return f'|{ZMat2str(x)}>'

#############################################################
## Code Analysis Tools
#############################################################


def nkdReport(SX,LX,SZ,LZ):
    '''Report [[n,k,dX,dZ]] for CSS code specified by SX,LX,SZ,LZ.'''
    k,n = np.shape(LX)
    LZ = minWeightLZ(SZ,LZ)
    dZ = min(np.sum(LZ,axis=-1))
    LX = minWeightLZ(SX,LX)
    dX = min(np.sum(LX,axis=-1))
    d = min([dX,dZ])
    gamma = codeGamma(n,k,d)
    return(f'n:{n} k: {k} dX: {dX} dZ:{dZ} gamma:{gamma}')

def codeGamma(n,k,d):
    if d > 1:
        return(np.log (n / k)/np.log(d))
    return 10

def tOrthogonal(SX,target=None):
    '''Return largest t for which the weight of the product of any t rows of SX is even.'''
    t = 1
    r, n = np.shape(SX)
    target = r if target is None else target
    while t + 1   <= target:
        ## all combinations of t+1 rows
        for ix in iter.combinations(range(r),t+1):
            T = np.prod(SX[ix,:],axis=0)
            if np.sum(T) % 2 == 1:
                return t 
        t += 1
    return t

##########################################################
## Check for quasi-transversality
##########################################################

def QuasiTransversal(SX,LX,d):
    '''Check for quasi transversality - see Pin Code paper'''
    ## intersections of up to d elements of SX
    SXE = SXIntersections(SX,d)
    ## check they are even weight
    for x in SXE[-1]:
        if (len(x) % 2) > 0:
            return False
    ## intersections of up to d-1 elements of LX
    LXE = SXIntersections(LX,d-1)
    for s in range(0,d-1):
        for t in range(0,d-1-s):
            for x1 in SXE[s]:
                for x2 in LXE[t]:
                    if (len(x1.intersection(x2)) % 2) > 0:
                        return False
    return True

def SXIntersections(SX,d):
    '''make products of up to d rows of SX and return list of sets'''
    r,n = np.shape(SX)
    E = []
    E.append([set(bin2Set(x)) for x in SX])
    for c in range(d-1):
        temp = set()
        for i in range(len(E[0])):
            for j in range(len(E[c])):
                e = E[0][i].intersection(E[c][j])
                if len(e) > 0 and len(e) < len(E[c][j]):
                    temp.add(set2tuple(e))
        E.append([set(e) for e in temp])
    return E



##################################################
## Action and Tests for Diagonal Logical Operators
##################################################

def isDiagLO(z,SX,K_M,N):
    '''Test if z is the Z-component of a logical XP operator.
    Inputs:
    z: Z_N vector to test
    SX: X-checks
    K_M: Z_N matrix representing phase and Z-components of diagonal logical identities
    N: precision of XP operators'''
    for x in SX:
        ## componentwise product of x and z
        xz = x * z 
        ## Z and phase component of commutator 
        czp = np.mod(np.hstack([-2*xz,[np.sum(xz)]] ),N)
        ## residue of czp with respect to logical identities
        c, v = HowRes(K_M, czp, N)
        ## if non-zero, there is an error
        if np.sum(c) > 0:
            print(func_name(),f'Error: x={x},z={z},c={c}')
            return False 
    return True

def CPIsLO(qVec,SX,CPKM,V,N,CP=True):
    '''Check if qVec is a logical operator by calculating the group commutator [[A, CP_V(qVec)]] for each A in SX
    Inputs: 
    qVec: Z_2N vector of length |V| representing a product of CP operators prod_{v in V}CP_N(qVec[v],v)
    SX: binary matrix representing X-checks
    CPKM: Z_2N Howell form matrix representing logical identities in CP form
    V: binary matrix representing vector part of CP operators
    N: precision of operators
    CP: if True, treat operators as CP operators, otherwise RP operators'''
    ## check that there is an all-zero row of V
    if pIndex(V) is None:
        r,n = np.shape(V)
        V = np.vstack([V,ZMatZeros(n)])
        qVec = np.hstack([qVec,[0]])
        r,n = np.shape(CPKM)
        CPKM = np.hstack([CPKM,ZMatZeros((r,1))])
    for x in SX: 
        c = CPComm(qVec,x,V,N,CP)
        c, u = HowRes(CPKM,c,2*N)
        c = np.mod(c,2*N) 
        if np.sum(c) != 0:
            print('x',x2Str(x))
            print('qVec',CP2Str(qVec,V,N))
            print('c',CP2Str(c,V,N))
            return False
    return True

def action2CP(vList,pVec,N):
    '''Return controlled phase operator corresponding to phase vector pVec
    pList a list of phases applied on each by an operator
    return a CP operator'''
    qVec = pVec.copy()
    for i in range(len(qVec)):
        v, p = vList[i],qVec[i]
        if p > 0:
            ix = uLEV(v,vList)
            qVec = np.mod(qVec-p*ix,N)
            qVec[i] = p
    return qVec

def DiagLOActions(LZ,LX,N):
    '''Return phase vector for each Z component in LZ.
    Inputs:
    LZ: Z_N matrix whose rows are the z-vectors
    LX: binary matrix - X-logicals
    N: precision of XP operators.
    Output: 
    phase vector p for each z component in LZ such that XP(0|0|z)|v>_L = w^{p[v]}|v>_L plus list of indices v
    '''
    t = log2int(N)
    k,n = LX.shape
    vLX,vList = Orbit2dist(LX,t,True)
    pVec = ZMat(np.mod([[np.dot(z,x) for x in vLX] for z in LZ],N))
    return pVec,vList

def codeword_test(qVec,SX,LX,V,N):
    '''Print report on action of prod_{v in V}CP_N(qVec[v],v) operator on comp basis elts in each codeword.
    Inputs:
    qVec: Z_2N vector of length |V| representing CP operator
    SX: X-checks in binary matrix form
    LX: X-logicals in binary matrix form
    V: vectors for CP operator
    N: precision of CP operator'''
    vList, CW = codewords(SX,LX)
    for i in range(len(CW)):
        s = {CPACT(e,qVec,V,N) for e in CW[i]}
        print(f'{ket(vList[i])}L {s}')

def action_test(qVec,LX,t,V):
    '''Return action of prod_{v in V}CP_N(qVec[v],v) operator on codewords in terms of CP operators.'''
    N = 1 << t
    ## test logical action
    Em,vList = Orbit2dist(LX,t,True)
    ## phases are modulo 2N 
    pList = ZMat([CPACT(e,qVec,V,N) for e in Em])//2
    act = action2CP(vList,pList, N)
    return 2*act, vList

def CZLO(SX, LX):
    '''Find logical operators made from T gates and CZ for triorthogonal codes'''
    t = 3
    # make Howell Form of SX and LX
    SXH = getH(SX, 2)
    LXH = getH(LX, 2)
    ## information set for SX,LX
    li = [leadingIndex(x) for x in SXH] + [leadingIndex(x) for x in LXH]
    k, n = np.shape(LXH)
    N = 1 << t
    ## only need to consider CZ between cols in info set
    V = [set2Bin(n,c) for c in iter.combinations(li,2)]
    ## this ordering reduces number of CZ
    V = np.vstack([V,ZMatI(n)])
    V = np.vstack([ZMatI(n),V])

    ## embedded code
    SX_V = matMul(SX,V.T,2)
    LX_V = matMul(LX,V.T,2)
    SX_V, LX_V, SZ_V, LZ_V = CSSCode(SX_V,LX_V)

    r_V,n_V = np.shape(SX_V)
    ## Logical identities
    if t == 1: 
        K_M = ZMatZeros((1,n_V+1))
    elif t == 2 and SZ_V is not None:
        K_M = np.hstack([SZ_V,ZMatZeros((len(SZ_V),1))]) * 2
    else:
        K_M = LIAlgorithm(LX_V,SX_V,N//2,debug=False) * 2
    ## z components of generators of diagonal XP LO group
    K_L = DiagLOComm(SX_V,K_M,N)
    v = ZMat([1] * n + [2] * (len(V)-n))
    K_L = nsIntersection([K_L,np.diag(v)],N)
    
    ## phase applied to codewords
    pList, VL = DiagLOActions(K_L,LX_V,N)
    ## actions as CP operators
    qList = ZMat([action2CP(VL,pVec,N) for pVec in pList])

    A = np.hstack([qList,K_L])
    A = getH(A,N)
    qList, zList = A[:,:len(VL)], A[:,len(VL):]
    for z,qL in zip(zList,qList):
        q, Vq = CP2RP(2*z,V,t,CP=False)
        if np.sum(qL) > 0:
            print(CP2Str(2*qL,VL,N),"=>",CP2Str(q,Vq,N))    
        # codeword_test(q,SX,LX,Vq,N)

########################################################
## Logical Identity Algorithm
########################################################

def LIAlgorithm(LX,SX,N,compact=True,debug=False):
    '''Run logical Identity Algorithm.
    Inputs:
    LX: X logicals
    SX: X-checks
    N: required precision
    compact: print full vector form if True, else support form
    debug: verbose output
    Output:
    KM: Z_N matrix representing phase and z components of diagonal logical identities.'''
    KM = getKM(SX, LX, N)
    if debug:
        print(f'\nLogical Identities Precision N={N}:')
        print(f'K_M = Ker_{N}(E_M):')
        if compact:
            print(ZMat2compStr(KM))
        else:
            print(ZMatPrint(KM))
    return KM

def getKM(SX, LX,N):
    '''Return KM - Z_N matrix representing phase and z components of diagonal logical identities.'''
    t = log2int(N)
    A = Orbit2dist(np.vstack([SX,LX]), t)
    A = np.hstack([A,[[1]]*len(A)])
    return getK(A,N)

########################################################
## Generators of Diagonal Logical XP Group via Kernel Method
########################################################

def diagLOKer(LX,SX,N,target=None):
    '''Return diagonal logical operators via Kernel method.
    Inputs:
    LX: X logicals
    SX: X-checks
    N: required precision
    target: if not None, search for an implementation of the target.
    Output:
    K_L: Z components and phase vectors of diagonal logical operators
    vList: list of codewords |v>_L which are the components of the phase vectors.'''
    r,n = np.shape(SX)
    ## t is orbit distance - if N = 2^t, use t else t is None
    t = log2int(N)
    ## Em are orbit reps 1-1 corresp with codewords
    ## only need to consider wt(v) <= t
    Em,vList = Orbit2dist(LX,t,True)
    if target is None:
        m = len(Em)
        ## delta: indicator vector for codeword
        bList = ZMatI(m)
    else:
        m = 1
        # indicator vector if target <= vList
        bList = uLEV(target,vList)
        bList = ZMat([bList]).T
    Em = np.hstack([Em,bList])
    ## Adding SX does not change delta
    SX = np.hstack([SX,ZMatZeros((r,m))])
    ## Apply SX up to orbit distance t
    E_0 = Orbit2dist(SX,t)
    E_L = np.vstack([np.mod(E_0 + e,2) for e in Em])
    ## calculate kernel modulo N
    K_L = getK(E_L,N)
    return K_L, vList

def ker_method(LX,SX,N,compact=True):
    '''Run the kernel method and print results.
    Inputs:
    LX: X logicals
    SX: X-checks
    N: required precision
    compact: if True, output full vector forms, otherwise support view.
    '''
    r,n = np.shape(SX)
    KL,V = diagLOKer(LX,SX,N)
    m = len(V)
    print(f'\nLogical Operators Precision N={N}:')
    print(f'K_L = Ker_{N}(E_L):')
    if compact:
        print(ZMat2compStr(KL))
    else:
        print(ZMatPrint(KL))

    print('\nLogical Operators and Phase Vectors:')
    print(f'z-component | p-vector')
    LZN, pList = KL[:,:n], np.mod(- KL[:,n:],N)
    KL = getH(np.hstack([pList, LZN]),N)
    for pz in KL:
        pvec, z = pz[:m], pz[m:]
        if np.sum(pvec) > 0:
            if compact:
                print(z2Str(z,N),":",row2compStr(pvec))
            else:
                print(ZMat2str(z),":",ZMat2str(pvec))

    print(f'\np-vector: p[i] represents phase of w^2p[i] on |v[i]> where v[i] is:')
    for i in range(len(V)):
        print(f'{i}: {ket(V[i])}')

########################################################
## Search for LO by logical action via Kernel Method
########################################################

def ker_search(target,LX,SX,t=None,debug=False):
    '''Run kernel search algorithm.
    Inputs:
    target: string corresponding logical CP operator to search for
    LX: X logicals
    SX: X-checks
    t: required level of Clifford hierarchy
    debug: if True, verbose output.
    Output:
    z-component of a diagonal XP operator implementing target, or None if this is not possible.
    '''
    r, n = np.shape(SX)
    k = len(LX)

    (x,qL), VL, t2 = Str2CP(target,n=k)
    if t is None:
        t = t2
    elif t > t2:
        qL = ZMat(qL) * (1 << (t-t2))
    N = 1 << t 
    SXLX = np.vstack([SX,LX])
    EL, uvList = Orbit2dist(SXLX,t,return_u = True)
    vList = uvList[:,r:]
    pList = np.mod(-ZMat([[CPACT(v,qL,VL,N)] for v in vList] )//2,N)
    KL = getK(np.hstack([pList, EL]),N)
    ## check if top lhs is 1 - in this case, the LO has been found
    if KL[0,0] == 1:
        z = KL[0,1:]
        if debug:
            pList, V = DiagLOActions([z],LX,N)
            q = action2CP(V,pList[0],N)
            print('operator:',z2Str(z,N))
            if n < 16:
                print(f'XP Form: XP_{N}(0|0|{ZMat2str(z)})')
            print("action:",CP2Str(2*q,V,N))
        return z
    if debug:
        print(func_name(),f'{target} Not found')
    return None

##########################################################
## Generators of Diagonal Logical XP Group via Commutator Method
##########################################################

def CPLogicalOps(SX,LX,K_M, N):
    '''Return list of z components of diagonal logical XP operators and their action in terms of CP operators.
    Inputs:
    LX: X logicals
    SX: X-checks
    K_M: phase and z components of diagonal logical XP identities 
    Output:
    LZ: list of z components
    CPList: list of q-vectors of CP operators
    vList: list of vectors v for the CP operators CP_N(q[v],v)'''
    LZ = DiagLOComm(SX,K_M,N)
    LA, vList = DiagLOActions(LZ,LX,N)

    a, b = np.shape(LA)
    A = np.hstack([LA,LZ])
    A = getH(A,N)
    LA, LZ = A[:,:b], A[:,b:]

    CPlist = ZMat([action2CP(vList,pList,N) for pList in LA])
    a, b = np.shape(CPlist)
    A = np.hstack([CPlist,LZ])
    A = getH(A,N)
    CPlist, LZ = A[:,:b], A[:,b:]
    return LZ, vList, CPlist


def DiagLOComm(SX,K_M,N):
    '''Return z components of generating set of logical XP operators using Commutator Method.
    Inputs:
    SX: X-checks
    K_M: phase and z components of diagonal logical XP identities 
    N: required precision
    Output:
    LZ: list of z components of logical XP operators.'''
    LZ = None
    for x in SX:
        Rx = commZ(x,K_M,N)
        LZ = Rx if LZ is None else nsIntersection([LZ, Rx],N)
    return LZ

def commZ(x,K_M,N):
    '''Return generating set of Z-components for which group commutator with X-check x is a logical identity.
    Inputs:
    x: an X-check (binary vector of length n)
    K_M: phase and z components of diagonal logical XP identities 
    N: required precision
    '''
    n = len(x)
    ## diag(x)
    xSet = bin2Set(x)
    if len(xSet) == 0:
        return ZMatI(n)
    ## rows of Ix are indicator vectors delta(x[i]==1)
    Ix = ZMat([set2Bin(n,[i]) for i in xSet],n)
    ## rows are of form (-2* delta(x[i]==1) | 1) because we require phase component equal to x.z 
    Rx = np.hstack([(N-2) * Ix,np.ones((len(Ix),1),dtype=int)])

    ## intersection with K_MS
    Rx = nsIntersection([K_M,Rx],N)
    # adjustment to ensure x.z = phase (last col)
    Rxp = np.sum(Rx[:,:-1],axis=-1) - 2* Rx[:,-1]

    Rx[:,xSet[-1]] += Rxp
    Rx = np.mod(Rx,2*N)
    ## solutions z are half the values in the intersection
    Rx = Rx[:,:-1]//2
    if N % 2 == 0 and np.sum(x) > 1:
        ## adding two elements N//2 to solutions z also meets requirements
        l = xSet[-1]
        Ix = Ix[:-1]
        Ix[:,l] = 1
        Rx = np.vstack([Rx,N // 2 * Ix])
    ## Where x[i] == 0, value of z is unrestricted
    Ix = ZMat([set2Bin(n,[i]) for i in bin2Set(1-x)],n)
    return np.vstack([Rx,Ix])

def comm_method(SX, LX, SZ, t, compact=True, debug=True):
    '''Run the commutator method and print results.
    Inputs:
    LX: X logicals
    SX: X-checks
    SZ: Z-checks
    t: Clifford hierarchy level
    compact: if True, output full vector forms, otherwise support view.
    debug: if True, verbose output.
    Output:
    zList: list of z-components generating non-trivial diagonal XP operators
    qList: list of q-vectors corresponding to logical action of each operator
    V: vectors indexing qList
    K_M: phase and z components of diagonal logical XP identities 
    '''
    r,n = np.shape(SX)
    N = 1 << t
    ## Logical identities
    if t == 1: 
        K_M = ZMatZeros((1,n+1))
    elif t == 2 and SZ is not None:
        K_M = np.hstack([SZ,ZMatZeros((len(SZ),1))]) * 2
    else:
        K_M = LIAlgorithm(LX,SX,N//2,compact,debug=debug) * 2
    ## z components of generators of diagonal XP LO group
    # K_L = DiagLOComm(SX,K_M,N)
    K_L = DiagLOComm_new(SX,K_M,N)

    ## phase applied to codewords
    pList, V = DiagLOActions(K_L,LX,N)
    ## actions as CP operators
    qList = ZMat([action2CP(V,pList,N) for pList in pList])

    if debug:
        print('\nApplying Commutator Method:')
        print('(z-component | q-vector | action)')
        for z, q in zip(K_L,qList):
            if np.sum(q) > 0:
                if compact:
                    print(z2Str(z,N),"|", row2compStr(q),"|", CP2Str(2*q,V,N))
                else:
                    print(ZMat2str(z), ZMat2str(q), CP2Str(2*q,V,N))        

        print(f'\nq-vector Represents CP_{N}(2q, w) where w =')
        for i in range(len(V)):
            print(f'{i}: {ZMat2str(V[i])}')
    ## Generators of logical actions
    A = np.hstack([qList,K_L])
    A = getH(A,N)
    qList, K_L = A[:,:len(V)], A[:,len(V):]

    ## update K_M
    ix = np.sum(qList,axis=-1) == 0 
    K_M = K_L[ix]
    ## non-trivial LO
    ix = np.sum(qList,axis=-1) > 0 
    K_L, qList = K_L[ix], qList[ix]

    if debug:    
        print('\nRearranging matrix to form (q, z) and calculating Howell Matrix form:')
        print('(z-component | q-vector | action)')
        for z, q in zip(K_L,qList):
            if compact:
                print(z2Str(z,N),"|", row2compStr(q),"|", CP2Str(2*q,V,N))
            else:
                print(ZMat2str(z), ZMat2str(q), CP2Str(2*q,V,N))     
    return K_L, qList, V, K_M


def DiagLOComm_new(SX,K_M,N):
    '''Return z components of generating set of logical XP operators using Commutator Method.
    Inputs:
    SX: X-checks
    K_M: phase and z components of diagonal logical XP identities 
    N: required precision
    Output:
    K_L: list of z components of logical XP operators.'''
    r,n = SX.shape
    ## partition X-checks into non-overlapping sets
    SXParts = SXPartition(SX)
    # print('SX Partitions',len(SXParts))
    K_L = None
    for XList in SXParts:
        ## find qubits which are not in the support of any of the X-checks in XList
        w = np.sum(XList,axis=0)
        ix = [i for i in range(n) if w[i] == 0]
        RX = ZMatZeros((len(ix),n))
        ## qubits in ix have no constraints for LZ
        for i in range(len(ix)):
            RX[i,ix[i]] = 1
        XConstr = [RX]
        ## Add constraints for each x in XList
        for x in XList:
            XConstr.append(commZ_new(x,K_M,N)) 
        XConstr = np.vstack(XConstr)
        ## Intersection with previous spans
        K_L = XConstr if K_L is None else nsIntersection([K_L,XConstr],N)
    return K_L

def commZ_new(x,K_M,N):
    '''Return generating set of Z-components for which group commutator with X-check x is a logical identity.
    Inputs:
    x: an X-check (binary vector of length n)
    K_M: phase and z components of diagonal logical XP identities 
    N: required precision
    '''
    n = len(x)
    ## get set bits
    xSet = bin2Set(x)
    ## number of set bits
    wx = len(xSet)
    ## rows are of form (-2 I | 1) because we require phase component equal to - x.z 
    RI = ZMatZeros((wx,n+1))
    for i in range(len(xSet)):
        RI[i,xSet[i]] = N-2
        RI[i,-1] = 1
    LI = nsIntersection([K_M,RI],N)
    # adjustment to ensure x.z = phase (last col)
    phaseAdj = np.sum(LI[:,:-1],axis=-1) - 2 * LI[:,-1]
    LI[:,xSet[-1]] = np.mod(LI[:,xSet[-1]] + phaseAdj,2*N)
    ## adding N in pairs on support of x also a solution
    RI = ZMatZeros((wx-1,n+1))
    for i in range(len(xSet)-1):
        RI[i,xSet[i]] = N
        RI[i,xSet[-1]] = N
    LI = np.vstack([LI,RI])
    ## solutions z are half the values in the intersection + drop phase component
    return LI[:,:-1] // 2

def SXPartition(SX):
    '''Partition SX into non-overlapping sets'''
    ## order SX via BFS search
    ix = BFSOrder(SX)
    SX = SX[ix]
    todo = set(range(len(SX)))
    temp = []
    while len(todo) > 0:
        ix = SXCC(SX,todo)
        temp.append(SX[sorted(ix)])
        todo = todo - ix
    return temp

def SXCC(SX,todo):
    '''Find largest set of checks which don't overlap with those in todo'''
    temp = set()
    while len(todo) > 0:
        ## add smallest index to temp
        i = min(todo)
        temp.add(i)
        ## update todo to include only X-checks with no overlap with SX[i]
        todo = nonOverlapping(SX,i,todo)
    return temp

def nonOverlapping(SX,i,todo):
    ix = bin2Set(SX[i])
    return [j for j in todo if np.sum(SX[j,ix])==0]

def edgeAdj(SX):
    '''Adjacency matrix for X-checks SX
    Checks are adjacent if separated by an edge'''
    r = len(SX)
    ## Edges are overlaps of two X-checks
    E = [SX[i]*SX[j] for (i,j) in iter.combinations(range(r),2)]
    E = ZMat([e for e in E if np.sum(e) > 0])
    ## Initialise adjacency matrix
    A = {i: set() for i in range(r)}
    ## Iterate through edges
    for i in range(len(E)):
        # EA = set()
        ix = bin2Set(E[i])
        wE = len(ix)
        ## overlap of each X-check with the edge
        wSX = np.sum(SX[:,ix],axis=-1) 
        ## checks are connected via the edge if they have non-zero overlap, but do not contain the edge
        EA = [i for i in range(len(wSX)) if wSX[i] > 0 and wSX[i] < wE]
        ## all pairs of checks in EA are considered adjacent - update adjacency matrix
        for (j,k) in iter.combinations(EA,2):
            A[j].add(k)
            A[k].add(j)
    return A

def BFSOrder(SX):
    '''traverse SX by overlap and record BFS ordering'''
    r,n = SX.shape 
    ## get adjacency matrix
    A2 = edgeAdj(SX)
    visited = set()
    tovisit = set(range(r))
    temp = ZMatZeros(r)
    ## BF search - keep track of the order in which checks are encountered
    count = 0
    ## There may be more than one connected component
    CCcount = 0
    while len(tovisit) > 0:
        CCcount += 1
        j = min(tovisit)
        tovisitloop = [j]
        tovisit.remove(j)
        visited.add(j)
        temp[count] = j
        count+=1
        while len(tovisitloop) > 0:
            i = tovisitloop.pop(0)
            for j in A2[i]:
                if j not in visited:
                    tovisitloop.append(j)
                    tovisit.remove(j)
                    visited.add(j)
                    temp[count] = j
                    count+=1
    # print(func_name(),'SX Connected Components',CCcount)
    return temp

############################################################
## Depth One Algorithm
############################################################

def depth_one_t(SX,LX,t=2,cList=None, debug=False):
    '''Run depth-one algorithm - search for transversal logical operator at level t of the Clifford hierarchy
    Inputs:
    LX: X logicals
    SX: X-checks
    t: required level of Clifford hierarchy
    cList: partition of qubits (optional) - improves runtime for known symmetries
    debug: if true, verbose output
    Output: if depth-one operator found:
    cList: partition of the qubits as a list of cycles of the qubits
    target: logical action as text representation of CP operator
    '''
    r,n = np.shape(SX)
    k,n = np.shape(LX)
    N = 1 << t

    if cList is None:
        ## All binary vectors of length n weight 1-t
        V = Mnt(n,t)
    else:
        ## V based on partition supplied to algorithm
        V = Mnt_partition(cList,n,t)

    ## move highest weight ops to left - more efficient in some cases
    V = np.flip(V,axis=0)
    ## Construct Embedded Code
    SX_V = matMul(SX, V.T, 2)
    LX_V = matMul(LX, V.T, 2)
    SZ_V = None
    ## Find diagonal logical operators at level t
    K_L, qList, VL, K_M = comm_method(SX_V, LX_V, SZ_V, t, compact=False, debug=False)

    ## zList includes both trivial and non-trivial LO
    # zList = np.vstack([K_L,K_M])
    ## Calculate clifford level of the logical operators
    tList = [CPlevel(2*qL,VL,N) for qL in qList]
    ## Calculate overlap (how many times a qubit is used in more than one operator) for level t operators
    ix = [i for i in range(len(tList)) if tList[i] == t]
    if len(ix) == 0:
        print(f'No level {t} logical operators found.')
        return None
    j = ix[0]
    ## convert to CP form
    CPKM = [CP2RP(2 * q,V,t,CP=False,Vto=V)[0] for q in K_M]
    CPKL = [CP2RP(2 * q,V,t,CP=False,Vto=V)[0] for q in K_L]

    ## get jth row - level t LO
    qCP = CPKL.pop(j)

    ## Howell form modulo 2N
    CPKL = ZMat(CPKL + CPKM)
    CPKL = getH(CPKL,2*N)
    if debug:
        target = CP2Str(2*qList[j],VL,N)
        print('Target - Logical action:',target)
        print('Target - CP Form',ZMat2str(qCP) )
        print(CP2Str(qCP,V,N,True))
    ## Search for depth-one operator 
    sol = findDepth1(CPKL,qCP,V,N)
    if sol is not False: 
        print("\nDepth-One Solution Found")
        qCP =  sol
        cList = CP2Partition(qCP,V)
        print('Qubit Partition', cList)
        op = CP2Str(qCP, V, N)
        print('Logical Operator:', op)       
        qRP = CP2RP(qCP,V,t,CP=True,Vto=V)[0]
        pList, VL = DiagLOActions([qRP//2],LX_V,N)
        qList = action2CP(VL,pList[0],N)
        target = CP2Str(qList*2,VL,N)
        print('Logical action:',target)
        CPKM = ZMat(CPKM)
        CPKM = getH(CPKM, 2*N)
        print('LO Test:',CPIsLO(qCP,SX,CPKM,V,N))
        if len(LX) < 5:
            print('Testing Action on Codewords')
            codeword_test(qCP,SX,LX,V,N)
        return cList, target
    else:
        print("\nNo Depth-One Solution Found")

############################################################
## Helper Functions for findDepth1
############################################################

def LR2ix(LR):
    ## return ix: a list of 3 lists. ix[i] is the list of indices j with LR[j] = i
    ix = [[] for i in range(3)]
    for i in range(len(LR)):
        ix[LR[i]].append(i)
    return ix

def CPACT(e,qVec,V,N): 
    ## return action of CP operator \prod_{v in V} CP_N(qVec[v],v)  on comp basis elt |e>
    xrow = np.abs(uGEV(e,V))
    return np.sum(xrow * qVec ) % (2*N)

def FDoverlap(j, V):
    '''return list of indices i in [0..|V|-1] not equal to j such that wt(V[i]V[j]) > 0'''
    E = np.sum(V[j] * V,axis=1)
    return [i for i in range(len(V)) if E[i] > 0 and i != j]

def overlapCount(qVec, V):
    '''Return oc: number of qubits with support > 1 in qVec, and oix: index of first operator with overlap'''
    ## restrict V to non-zero elements of qVec
    Vix = bin2Set(qVec)
    V1 = V[Vix]
    ## for each qubit, how many operators act on it?
    vcount = np.sum(V1,axis=0)
    qix = [i for i in range(len(vcount)) if vcount[i] > 1]
    ## count of qubits which have more than one operator
    oc = len(qix)
    return oc

def findDepth1(KL,qVec,V,N):
    '''Run Depth-one algorithm. 
    Inputs:
    KL: Z_2N matrix representing logical identities and operators of the embedded code
    qVec: Z_2N vector representing a logical operator we want to find a depth-one implementation for
    V: Embedding matrix - tells us which physical qubits are involved in each CP/RP operator
    N: required precision
    Output:
    False if no depth one operator found, otherwise z-component of the embedded operator'''
    ## set of configurations already considered
    visited = set()
    ## configurations to test - initial config is all in centre
    todo = [tuple([1]*len(V))]
    while len(todo) > 0:
        ## LR[i] = 0: moved to the left to be eliminated
        ## LR[i] = 2: moved to the right to be retained
        ## LR[i] = 1: in the centre - to be assigned a side
        ## get next configurations
        LR = todo.pop()
        ## get indices of left/centre/right
        LRix = LR2ix(LR)
        ## ix is a flat array
        ix = [a for r in LRix for a in r]
        ## z1 is the residual 
        qVec1, u = HowRes(KL[:,ix], qVec[ix],2*N)
        ## s is the support of z1
        s = bin2Set(qVec1)
        m = min(s)
        ## if m < LR[0] then it was not eliminated - invalid configuration
        if m >= len(LRix[0]): 
            ## indices corresp to original col order
            ixR = ixRev(ix)

            ## reorder z1
            qVec1 = qVec1[ixR] 
            oc = overlapCount(qVec1, V)
            # ## original order of m
            oix = ix[m] 

            ## test for depth-one operator:
            if oc == 0:
                return qVec1
            ## get overlap with m
            ix = FDoverlap(oix, V) 
            ## LR1: overlap with m moved to left; m moved to the right
            LR1 = list(LR) 
            for i in ix:
                LR1[i] = 0    
            LR1[oix] = 2
            ## LR: m moved to left
            LR2 = list(LR)
            LR2[oix] = 0
            for A in LR2, LR1:
                A = tuple(A)
                ## add to visited and todo if not already encountered
                if A not in visited:
                    visited.add(A) 
                    todo.append(A)
    ## we have looked at all possible configurations with no valid solution
    return False


############################################################
## Canonical Logical Operators
############################################################

def canonical_logical_algorithm(q1,V1,LZ,t,Vto=None,debug=True):
    '''Run canonical logical operator algorithm for given code and target CP operator
    Inputs:
    q1: q-vector for CP operator prod_{v in V1}CP_N(q[v],v)
    V1: support vectors for CP operator
    LZ: binary matrix representing Z-logicals 
    t: Clifford hierarchy level
    Vto: if not None, return CP operators using vectors from Vto
    debug: if True, verbose output'''
    q1, V1 = CPNonZero(q1,V1)
    N = 1 << t
    ## q1, V1 are logical CP operators - convert to RP
    q2, V2 = CP2RP(q1,V1,t,CP=True,Vto=None)
    ## canonical logical in RP form
    q3 = q2 
    V3 = matMul(V2,LZ,2) 
    ## convert to CP - this will ensure max support size of operators is t
    q4,V4 = CP2RP(q3,V3,t,CP=False,Vto=Vto)

    q5,V5 = CP2RP(q4,V4,t,CP=True,Vto=Vto)
    if debug:
        print('Canonical Logical Operator Implementation - CP')
        print(CP2Str(q4,V4,N,CP=True))
        print('Canonical Logical Operator Implementation - RP')
        print(CP2Str(q5,V5,N,CP=False))
    return (q4,V4), (q5,V5)

############################################################
## Generate Codes with Desired Logical Operators
############################################################

def CSSPuncture(SX,LX,z):
    '''z is an LO for CSS code SX/LX. Remove qubits where z[i] = 0, and make SX/LX into a nice format'''
    ## ix are non-zero entries of z
    ix = bin2Set(z)
    n = len(ix)
    ## extract columns indexed by ix, and calc Howell form
    SX = getH(SX[:,ix],2)
    LX = getH(LX[:,ix],2)
    ## sort cols of SX/LX by weight of SX then alpha SX then weight LX then alpha LX
    cList = [(sum(SX[:,i]),tuple(SX[:,i]),sum(LX[:,i]),tuple(LX[:,i])) for i in range(n)]
    ix = argsort(cList)
    ## Flip row order
    LX = np.flip(LX[:,ix],axis=0)
    SX = np.flip(SX[:,ix],axis=0)
    return SX,LX


# def CSSwithLO_old(target,d):
#     '''Return a CSS code with an implementation of a desired logical CP operator using single-qubit phase gates.
#     Inputs:
#     target: string representing the target CP operator
#     d: distance of the toric code the resulting code is based on.
#     Output:
#     SX, LX and Clifford hierarchy level'''
#     (x, qL), VL, t = Str2CP(target)
#     k = len(VL[0])
#     N = 1 << t
#     SX,SZ = toricDd(k,d)
#     SX,LX,SZ,LZ = CSSCode(SX,SZ=SZ,simplifyGens=True)
#     ## Get Canonical LO
#     (qCP,VCP), (qRP,V) = canonical_logical_algorithm(qL,VL,LZ,t,debug=False)

#     ## sort V so that cols with lowest gcd are to the bottom
#     z = qRP//2
#     g = np.gcd(z,N)
#     maxg = max(np.mod(g,N))
#     if maxg > 1:
#         ix = argsort(list(zip(g,z)),reverse=True)
#         V = V[ix]
#         z = z[ix]
#     ## make embedded code
#     SX = matMul(SX,V.T,2)
#     LX = matMul(LX,V.T,2)
#     ## z component of transversal LO on embedded code
#     print(func_name(),ZMat2str(z))
    
#     ## simplify z component
#     if maxg > 1 and t > 1:
#         K_M = getKM(SX,LX,N//2)[:,:-1] * 2
#         z, u = HowRes(K_M, z,N)
#     ## remove qubits where z[i] = 0, and make SX/LX into a nice format
#     SX,LX = CSSPuncture(SX,LX,z)
#     return SX,LX


# def CSSwithLO_old(target,d):
#     '''Return a CSS code with an implementation of a desired logical CP operator using single-qubit phase gates.
#     Inputs:
#     target: string representing the target CP operator
#     d: distance of the toric code the resulting code is based on.
#     Output:
#     SX, LX and Clifford hierarchy level'''
#     (x, qL), VL, t = Str2CP(target)
#     k = len(VL[0])
#     N = 1 << t
#     SX,SZ = toricDd(k,d)
#     SX,LX,SZ,LZ = CSSCode(SX,SZ=SZ,simplifyGens=True)
#     ## Get Canonical LO
#     (qCP,VCP), (qRP,V) = canonical_logical_algorithm(qL,VL,LZ,t,debug=False)
#     ##v sort V so that cols with lowest gcd are to the bottom
#     z = qRP//2
#     g = np.gcd(z,N)
#     ix = (g == 1)
#     ## discard qubits with gcd > 1 wrt N
#     V = V[ix]
#     ## make embedded code
#     SX = matMul(SX,V.T,2)
#     LX = matMul(LX,V.T,2)
#     ## simplify rep of SX/LX
#     R,V,H,U,K = solveHU(SX,LX,2)
#     return H,R

def CSSwithLO(target,d):
    '''Return a CSS code with an implementation of a desired logical CP operator using single-qubit phase gates.
    Inputs:
    target: string representing the target CP operator
    d: distance of the toric code the resulting code is based on.
    Output:
    SX, LX and Clifford hierarchy level'''
    (x, qL), VL, t = Str2CP(target)
    k = len(VL[0])
    N = 1 << t
    SX,SZ = toricDd(k,d)
    SX,LX,SZ,LZ = CSSCode(SX,SZ=SZ,simplifyGens=True)
    ## Get Canonical LO
    (qCP,VCP), (qRP,V) = canonical_logical_algorithm(qL,VL,LZ,t,debug=False)
    ##v sort V so that cols with lowest gcd are to the bottom
    z = qRP//2
    g = np.gcd(z,N)
    ix = argsort(g,reverse=True)
    V = V[ix]
    ## make embedded code
    SX_V = matMul(SX,V.T,2)
    LX_V = matMul(LX,V.T,2)
    LX,v,SX,U,K = solveHU(SX_V,LX_V,2)
    z = ker_search(target,LX,SX,t)
    # ## discard qubits with z == 0
    ix = z > 0
    z = z[ix]
    SX = SX[:,ix]
    LX = LX[:,ix]
    return SX, LX, z
    # V = V[ix]
    # SX = matMul(SX,V.T,2)
    # LX = matMul(LX,V.T,2)
    ## simplify rep of SX/LX
    # R,V,H,U,K = HowSolve(SX,LX,2)
    # return H,R,z

def codeSearch(target, d, debug=False):
    '''Run the algorithm for generating a CSS code with transversal implementation of a desired logical operator and print output
    Inputs:
    target: string representing the target CP operator
    d: distance of the toric code the resulting code is based on.
    debug: if true, verbose output'''
    ## make a CSS code
    SX,LX,LO = CSSwithLO(target,d)
    # LO = ker_search(target,LX,SX)
    SX,LX,SZ,LZ = CSSCode(SX,LX)
    ## calculate distance
    LZ = minWeightLZ(SZ,LZ,method='mw')
    dZ = np.min(np.sum(LZ,axis=-1))
    LX = minWeightLZ(SX,LX,method='mw')
    dX = np.min(np.sum(LX,axis=-1))
    return SX,LX,SZ,LZ,dX,dZ,LO

######################################
## Local Logical Operators
######################################

def tValentProduct(SX,t):
    '''Products of t different rows from SX'''
    temp = set()
    r, n = np.shape(SX)
    for s in iter.combinations(range(r),t):
        x = np.product(SX[list(s)],axis=0)
        if np.sum(x) > 0:
            temp.add(tuple(x))
    return ZMat(temp)

def VLocal(SX,t):
    '''Return list of groups of up to t qubits connected by a row of SX'''
    SXSets = [bin2Set(x) for x in SX]
    return {c for x in SXSets for s in range(2,t+1) for c in iter.combinations(x, s) }

def LocalLO(SX, LX, SXLX, t):
    '''Look for local logical operators - ie those formed from single and multiqubit gates
    but where multiqubit gates only can be applied to 'local' qubits specified by SXLX'''
    k, n = np.shape(LX)
    N = 1 << t
    print('\nEMBEDDED CODE')
    print('Neighbour Graph Edge Weights',set(np.sum(SXLX,axis=-1)))

    V = VLocal(SXLX,2)
    print('Extra Qubits for Embedded Code', len(V))
    V = [set2Bin(n,v) for v in V]
    V = np.vstack([ZMatI(n),V])
    SX_V = matMul(SX,V.T,2)
    LX_V = matMul(LX,V.T,2)

    SX_V, LX_V, SZ_V, LZ_V = CSSCode(SX_V,LX_V)

    zList,qList, VL, K_M = comm_method(SX_V, LX_V, SZ_V, t,debug=False)
    
    print('\nLocal Diagonal Logical Operators via Embedded Code')
    for z,qL in zip(zList,qList):
        q, Vq = CP2RP(2*z,V,t,CP=False)
        print(CP2Str(2*qL,VL,N),"=>",CP2Str(q,Vq,N)) 