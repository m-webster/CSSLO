import numpy as np
from common import *
import numba as nb
import cProfile
import tracemalloc
# from distance import *


#####################################################
## Ring Functions for Calculating Howell Matrix Form
#####################################################

@nb.jit(nb.int16[:,:](nb.int16,nb.int16,nb.int16))
def Gcdex_jit(a,b,N):
    '''Extended GCD: Return g,s,t,u,v such that: as + bt = g where g = gcd(a,b); AND au + bv = 0'''
    a = a % N
    b = b % N
    s = nb.int16(0)
    old_s = nb.int16(1)
    t = nb.int16(1)
    old_t = nb.int16(0)
    r = b
    old_r = a
    while r != nb.int16(0):
        quotient = old_r // r
        (old_r, r) = (r, old_r - quotient * r)
        (old_s, s) = (s, old_s - quotient * s)
        (old_t, t) = (t, old_t - quotient * t)
    p = np.sign(t * old_s - s * old_t)
    u = p * s
    v = p * t
    # g = old_r
    s = old_s
    t = old_t
    return np.array([[s,t],[u,v]],dtype=nb.int16)

@nb.jit(nb.int16(nb.int16,nb.int16))
def Ann_jit(a,N):
    '''Annihilator of a modulo N: Return u such that a u mod N = 0. Return 0 if a is a unit. Return 1 if a == 0'''
    a = a % N
    if a == 0:
        return 1
    u = N // np.gcd(a,N)
    return u % N

@nb.jit(nb.int16(nb.int16,nb.int16))
def Split_jit(a,N):
    a = a % N
    if N == nb.int16(0):
        return nb.int16(0)
    if a == nb.int16(0):
        return nb.int16(1)
    r = int(np.ceil(np.log2(np.log2(N)))) if N > 1 else nb.int16(1)
    for i in range(r):
        a = a*a % N
    return N // np.gcd(a,N)

@nb.jit(nb.int16(nb.int16,nb.int16,nb.int16))
def Stab_jit(a,b,N):
    ### return c such that GCD(a + bc, N) = GCD(a,b) modulo N
    a = nb.int16(a % N)
    b = nb.int16(b % N)
    g = np.gcd(np.gcd(a,b),N)
    c = Split_jit(a//g,N//g)
    return c % N

@nb.jit(nb.int16(nb.int16,nb.int16,nb.int16))
def Div_jit(a,b,N):
    '''Return c such that bc = a mod N or None if no such c exists'''
    a = a % N
    b = b % N
    if b < 1:
        return nb.int16(-1)
    g = np.gcd(b,N)
    if a % g == nb.int16(0):
        r = a % b
        while r > nb.int16(0):
            a += N
            r = a % b
        return a // b % N
    return nb.int16(-1)

@nb.jit(nb.int16(nb.int16,nb.int16))
def Unit_jit(a,N):
    '''Return a unit c such that ac = gcd(a,N)  mod N.'''
    a = a % N
    if a == nb.int16(0):
        return nb.int16(1)
    g = np.gcd(a,N)
    s = Div_jit(g,a,N)
    if g == nb.int16(1):
        return s
    d = Stab_jit(s,N//g,N)
    c = (s + d * N // g) % N
    return c



#####################################################
## RREF Modulo 2
#####################################################

@nb.jit(nb.types.Tuple((nb.int8[:,:],nb.int64[:]))(nb.int8[:,:],nb.int64,nb.int64,nb.int64,nb.int64))
def HowZ2(A,tB,nB,nC,r0):
    pivots = []
    B = A.copy()
    if np.sum(A) == 0:
        return B,np.array(pivots,dtype=nb.int64)
    m = len(B)
    r = r0

    for j in range(nC):
        for t in range(tB):
            ## c is the column of B we are currently looking at
            c = j + t * nB
            iList = [i for i in range(r,m) if B[i,c] > 0]
            if len(iList) > 0:
                i = iList.pop(0)
                pivots.append(c)
                ## found j: if j > r, swap row j with row r
                if i > r:  
                    ## swap using bitflips - more elegant than array indexing
                    B[r] = B[r] ^ B[i]
                    B[i] = B[r] ^ B[i]
                    B[r] = B[r] ^ B[i]
                ## eliminate non-zero entries in column c apart from row r
                for i in [i for i in range(r) if B[i,c] > 0] + iList:     
                    B[i] = B[i] ^ B[r]
                r +=1
    return B,np.array(pivots,dtype=nb.int64)

#####################################################
## Calculate Howell Matrix form modulo N
#####################################################

@nb.jit(nb.types.Tuple((nb.int16[:,:],nb.types.List(nb.int64)))(nb.int16[:,:],nb.int16,nb.int64,nb.int64,nb.int64,nb.int64))
def HowZN(A,N,tB,nB,nC,r0):
    '''Return Howell basis of A mod N plus row operations to convert to this form'''
    pivots = []
    if np.sum(A) == 0:
        return A,pivots
    m,n = A.shape
    B = [a for a in A]
    N = nb.int16(N)
    r = r0

    ## c is the column of B we are currently looking at
    for mc in range(nC):
        for t in range(tB):
            c = mc + t * nB
            ## find j such that B[j][c] > 0
            jList = [j for j in range(r,m) if B[j][c] > 0]
            if len(jList) > 0:
                j = jList.pop(0)
                pivots.append(mc)
                ## found j: if j > r, swap row j with row r
                if j > r:  
                    B[j],B[r] = B[r],B[j]
                ## Multiplying by x ensures that B[r][c] is a minimal representative
                b = B[r][c]
                x = Unit_jit(b,N)
                if(x > 1):
                    B[r] = np.mod(B[r] * x, N)
                ## eliminate entries in column c below row r
                for j in jList:
                    a, b = B[r][c],B[j][c]
                    C = Gcdex_jit(a,b,N)
                    Br = np.mod(B[r] * C[0,0] + B[j] * C[0,1], N)
                    Bj = np.mod(B[r] * C[1,0] + B[j] * C[1,1], N)
                    B[r] = Br
                    B[j] = Bj
                ## ensure entries in column c above row r are less than B[r][c]
                b = B[r][c]
                for j in range(r):
                    if B[j][c] >= b:
                        x = nb.int16(B[j][c] // b)
                        B[j] = np.mod(B[j] - B[r] * x,N)
                ## Multiplying by x = Ann(b) eliminates b = B[r][c], but rest of the row may be non-zero
                ## If x > 0 then b is a zero divisor and we add a row
                ## If x == 0, b is a unit and we move to the next value of l
                x = Ann_jit(b,N)
                if x > 0:
                    B.append(np.mod(B[r] * x,N))
                    m += 1
                r +=1
    temp = np.empty((m,n),dtype=nb.int16)
    for i in range (m):
        temp[i] = B[i]
    return temp,pivots

# @nb.jit
def blockDims(n,nA=0,tB=1,nC=-1):
    nA = min(n,nA)
    nB = (n - nA) // tB
    if nC < 0 or nC > nB:
        nC = nB 
    return nA,nB,nC



def getH(A,N,nA=0,tB=1,nC=-1,r0=0,retPivots=False):
    '''Return Howell matrix form modulo N:
    A: input matrix
    N: linear algebra modulo N
    nA: number of cols appended to right - not subject to row reduction
    tB: number of blocks in the matrix
    nC: number of columns to reduce
    r0: starting row for reduction'''
    ## nB: number of columns in each block
    m,n = A.shape
    nA,nB,nC = blockDims(n,nA,tB,nC)
    if N==2:
        A = np.array(A,dtype=np.int8)
        H,pivots = HowZ2(A,tB,nB,nC,r0)
    else:
        A = np.array(A,dtype=np.int16)
        H,pivots = HowZN(A,N,tB,nB,nC,r0)
    w = np.sum(H,axis=-1)
    ix = [i for i in range(len(H)) if i < r0 or w[i] >0]
    H = H[ix]
    return (H, list(pivots)) if retPivots else H
    
def getHU(A,N,nA=0,tB=1,nC=-1,r0=0):
    '''Return Howell matrix form modulo N plus transformation matrix U such that H = U @ A mod N'''
    m,n = A.shape
    B = np.hstack([A,ZMatI(m)])
    nA += m
    HU = getH(B,N,nA,tB,nC,r0)
    return HU[:,:n],HU[:,n:]

def getK(A,N,nA=0,tB=1,nC=-1):
    '''Return Kernel K such that K @ A.T = 0 mod N'''
    ## Transpose so nA,tB,nC,r0 not applicable
    H, U = getHU(A.T,N)
    ix = np.sum(H,axis=-1) == 0
    K = U[ix,:]
    ## r0 not applicable...
    K = getH(K,N,nA,tB,nC)
    return K

def HowRes(A,B,N,tB=1):
    '''Return R, V such that  B = R + V @ A mod N for A,B matrices, unknowns R, V'''
    return solveH(A,B,N,tB)

def HowResU(A,B,N,tB=1):
    '''Return R, V such that  B = R + V @ A mod N for A,B matrices, unknowns R, V'''
    R,V,H,U,K = solveHU(A,B,N,tB)
    return R,V

def solveHU(A,B,N,tB=1):
    '''Solve B = R + (V + <K>) @ A mod N for A,B matrices, unknowns R, V, K
    A: m x n
    B: r x n
    R: r x n
    V: r x m
    K: k x m'''
    B1D = len(B.shape) == 1
    A, B = ZMat2D(A), ZMat2D(B)
    m,n = A.shape
    r,n1 = B.shape
    if n != n1:
        print(func_name(), 'A,B incompatible shape')
    ## Make matrix of form [[I,B,0],[0,A,I]]
    BA = np.hstack([B,ZMatZeros((r,m))])
    BA = np.vstack([BA,np.hstack([A,ZMatI(m)])])
    ## Howell form - only consider first m2 + n1 columns
    HBA = getH(BA,N,nA=m,tB=tB,r0=r)
    ## How results in matrix of form [[I,R,V],[0,H,U],[0,0,K]]
    ## R: top middle block of HBA
    R = HBA[:r,:n]
    ## if B is 1D, return 1D result
    if B1D:
        R = R[0]
    ## V: top right block of HBA - negative to reverse residue calc
    V = np.mod(-HBA[:r,n:],N)
    ## H: bottom middle block of HBA
    H = HBA[r:,:n]
    ## U: bottom right block of HBA
    U = HBA[r:,n:]
    ix = np.sum(H,axis=-1) > 0
    H,U,K = H[ix,:],U[ix,:],U[np.logical_not(ix), :]
    return R,V,H,U,K

def solveH(A,B,N,tB=1):
    '''Solve B = R + V @ A mod N for unknown R, H = getH(A,N)
    A: m x n
    B: r x n
    R: r x n'''
    B1D = len(np.shape(B)) == 1
    A, B = ZMat2D(A), ZMat2D(B)
    m,n = A.shape
    r,n1 = B.shape
    if n != n1:
        print(func_name(), 'A,B incompatible shape')
    ## Make matrix of form [[B],[A]]
    BA = np.vstack([B,A])
    ## Howell form - only consider first m2 + n1 columns
    HBA = getH(BA,N,tB=tB,r0=r)
    ## How results in matrix of form [[I,R,V],[0,H,U],[0,0,K]]
    ## R: top block
    R = HBA[:r,:]
    ## H: lower block
    H = HBA[r:,:]
    ## if B is 1D, return 1D result
    if B1D:
        R = R[0]
    return R,H

# @nb.jit
def HowPivots(A,nA=0,tB=1,nC=-1,r0=0):
    '''Leading indices of matrix in Howell/RREF form
    - Examine first nC columns
    - Matrix has nBlocks blocks A = [A_0|A_1|..|A_n-1]'''
    temp = []
    if np.sum(A) == 0:
        return temp
    m,n = A.shape
    nA,nB,nC = blockDims(n,nA,tB,nC)
    r,c = r0,0
    for r in range(m):
        while c < nC and np.all([A[r,c + t * nB] == 0 for t in range(tB)]):
            c+=1
        if c < nC:
            temp.append(c)
        else:
            return temp
    return temp

# def ZMatBlockSum(A,nA=0,tB=1,nC=-1,N=2):
#     '''Sum block of Zmat
#     - up to nC
#     - tB - number of blocks
#     - if A = [A_0|A_1|..|A_n] return A_0 + 2 A_1 + ... + 2^n A_n'''
#     nB,nC = blockDims(A,nA,tB,nC)
#     ix = ZMat(range(nC))
#     S = A.take(indices=ix, axis=-1)
#     for t in range(1,tB):
#         ix += nB
#         S += A.take(indices=ix, axis=-1) * (N ** t)
#     return S

# @nb.jit
# @nb.jit(nb.types.Tuple((nb.int8[:,:],nb.int64[:]))(nb.int8[:,:],nb.int64,nb.int64,nb.int64,nb.int64))

# @nb.jit
def ZMatTake(A,ix):
    '''Take indices ix form A along axis -1 '''
    B = A[:,ix].copy()
    return B

# @nb.jit
def ZMatBlockSum(A,nA=0,tB=1,nC=-1,N=2):
    '''Sum block of Zmat
    - up to nC
    - tB - number of blocks
    - if A = [A_0|A_1|..|A_n] return A_0 + 2 A_1 + ... + 2^n A_n'''
    m,n = A.shape
    nA,nB,nC = blockDims(n,nA,tB,nC)
    ix = np.arange(nC)
    S = ZMatTake(A,ix)
    for t in np.arange(1,tB):
        ix += nB
        S += ZMatTake(A,ix) * (N ** t)
    return S

def ZMatPermuteCols(A,ix,nA=0,tB=1,nC=-1):
    m,n = A.shape
    nA,nB,nC = blockDims(n,nA=nA,tB=tB,nC=nC)
    ix = list(ix) + invRange(nB,ix)
    ix = ZMat(ix)
    ind = np.arange(n)
    for t in range(tB):
        ind[t*nB:(t+1)*nB] = ix
        ix += nB
    return ZMatTake(A,ind)

####################################
## Matrix Multiplication - avoid overflow for large matrices
####################################

# @nb.jit (nb.int16[:,:](nb.int16[:,:],nb.int16[:,:],nb.int64))
# def matMulZN(A,B,N):
#     m1,n1 = A.shape
#     m2,n2 = B.shape
#     C = np.zeros((m1,n2),dtype=nb.int16)
#     m = min(n1,m2)
#     for i in range(m1):
#         for j in range(n2):
#             temp = nb.int16(0)
#             for k in range(m):
#                 temp = np.mod(temp + (A[i,k] * B[k,j]),N)
#             C[i,j] = temp
#     return C

def matMulZN(A,B,N):
    return np.mod(A @ B, N)

@nb.jit (nb.int8[:,:](nb.int8[:,:],nb.int8[:,:]))
def matMulZ2(A,B):
    m1,n1 = A.shape
    m2,n2 = B.shape
    C = np.zeros((m1,n2),dtype=nb.int8)
    m = min(n1,m2)
    for i in range(m1):
        for j in range(n2):
            temp = nb.int8(0)
            for k in range(m):
                temp = temp ^ (A[i,k] & B[k,j])
            C[i,j] = temp
    return C

def matMul(A,B,N):
    '''Multiply two integer matrices modulo N'''
    A = ZMat2D(A)
    B = ZMat2D(B)
    if N==2:
        A = np.array(A,dtype=np.int8)
        B = np.array(B,dtype=np.int8)
        return matMulZ2(A,B)
    else:
        A = np.array(A,dtype=np.int16)
        B = np.array(B,dtype=np.int16)
        return matMulZN(A,B,N)
    return np.mod(A @ B, N)

def mod1(A):
    '''Replace values > 0 in A with 1'''
    A = ZMat(A)
    A[A>0] = 1
    return A

def RemoveZeroRows(A,N=False):
    '''Remove any zero rows from integer matrix A'''
    A = ZMat(A)
    w = np.sum(A, axis=-1)
    return A[w > 0]

def ZMatWeight(A,nA=0,tB=1,nC=-1):
    return np.sum(mod1(ZMatBlockSum(A,nA,tB,nC,1)),axis=-1)

def pListDefault(tB=1,pI=0.7):
    pLen = (1 << tB) - 1
    p = (1-pI)/pLen
    return np.array([pI] + [p] * pLen)

def ZMatProb(A,pList,nA=0,tB=1,nC=-1,N=2):
    pList = np.array(pList)
    W = ZMatBlockSum(A,nA=nA,tB=tB,nC=nC,N=N)
    return np.product(pList[W],axis=-1)

def lowWeightGens(L,S=None,N=2,tB=1,pList=None,retProb=False):
    '''Get set of lowest weight generators of logicals L
    S is a list of stabilisers
    tB is number of blocks
    N is precision
    pList is list of probabilities'''
    # Default probabilities
    if pList is None:
        pList = pListDefault(tB=tB)
    w = ZMatProb(L,pList,tB=tB,N=N)
    ## order L by increasing prob/decreasing weight
    ix = argsort(w)
    L = L[ix]
    w = w[ix]
    LS = L if S is None else np.vstack([L,S])
    ## RREF plus transformation matrix
    H, U = getHU(LS,N,tB=tB)
    ## K is a list of linear combinations of rows which result in zero
    ix = np.sum(H,axis=-1) == 0
    K = U[ix,:]
    ## RREF - so combinations of lowest weight rows are to the RHS
    K, LI = getH(K,N,retPivots=True)
    # K = K[:,:len(L)]
    ## get leading entries of K - these correspond to rows we should exclude 
    ## as adding them to the set results in a redundancy
    # LI = HowPivots(K,nC=len(L))
    ix = invRange(len(L),LI)
    L = L[ix,:]
    w = w[ix]
    return (L,w) if retProb else L

##################################################
####            Operations on Spans           ####
##################################################

def affineIntersection(A1,o1,A2,o2,N,C=False):
    '''Calculate intersection of two affine spaces
    U1 = o1 + <A1> mod N
    U2 = o2 + <A2> mod N'''
    if C is not False:
        A,o = C
        tocheck = np.mod(o + A,N)
        tocheck = np.vstack([[o],tocheck])
        for v in tocheck:
            v1 = np.mod(v - o1,N)
            b,u = HowRes(A1,v1,N)
            if not isZero(b):
                print(v1, 'Not in span of A1')
                return False
            v2 = np.mod(v - o2,N)
            b,u = HowRes(A2,v2,N)
            if not isZero(b):
                print(v2, 'Not in span of A2')
                return False
        return True
    o = affineIntercept(A1,o1,A2,o2,N)
    if o is False:
        return False
    ## new affine space is intersection
    A = nsIntersection([A1,A2],N)
    ## residue mod A to simplify result
    R, V = HowRes(A, o, N)
    return A,R[0]

def affineIntercept(A1,o1,A2,o2,N,C=False):
    '''Calculate intersection of two affine spaces
    U1 = o1 + <A1> mod N
    U2 = o2 + <A2> mod N
    return o which is in both U1 and U2 or False'''
    if C is not False:
        ## check if C-o1 is in <A1>
        R,V = HowRes(A1,np.mod(C-o1,N),N)
        if not isZero(R):
            return False
        ## check if C-o2 is in <A2>
        R,V = HowRes(A2,np.mod(C-o2,N),N)
        if not isZero(R):
            return False
        return True
    A = np.vstack([A1, A2])
    ## find solution o1-o2 = (v1|v2) @ (A1//A2) = v1@A1 + v2@A2 mod N <=> o1 - v1@A1 = o2 + v2@A2
    R, v1v2 = HowResU(A, np.mod(o1-o2,N), N)
    if not isZero(R):
        ## there is no solution if residue is non-zero
        return False
    ## extract v1 from v1v2 - corresponds to first len(A1) entries of the vector
    v1 = v1v2[:,:len(A1)]
    v1A1 = matMul(v1, A1, N)[0]
    ## note minus sign
    return np.mod(o1 - v1A1,N)

def nsIntersection(Alist,N,C=False):
    '''Intersection of multiple rowspans AList modulo N.'''
    if C is not False:
        for A in Alist:
            ## check if C in <A> for each A in AList
            R,V = HowRes(A,C,N)
            if not isZero(R):
                return False
        return True
    if len(Alist) == 0:
        return False
    A = Alist[0]
    for B in Alist[1:]:
        if len(B) < len(A):
            A,B = B,A
        AB = np.vstack([A,B])
        Kt = getK(AB.T,N)
        A0 = np.vstack([A,ZMatZeros(B.shape)])
        C = matMul(Kt, A0,N)
        A = getH(C,N)
    return A

def nsUnion(Alist,N,C=False):
    '''Union of multiple rowspaces Alist modulo N'''
    if C is not False:
        ## check if A in C for all A in Alist
        for A in Alist:
            R,V = HowRes(C,A,N)
            if not isZero(R):
                return False
        return True
    if len(Alist) == 0:
        return False
    return getH(np.vstack(Alist),N) 

