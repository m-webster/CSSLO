from common import *
import itertools as iter
from NSpace import *
from XCP_algebra import *

##########################################
## CSS Codes
##########################################

## create CSS code from various input types
## Default is to give values for SX and LX, then calculate SZ and LZ
## Can take either text or arrays as input
def CSSCode(SX,LX=None,SZ=None,LZ=None):
    ## convert to ZMat format - can be either string or array
    SX = bin2Zmat(SX)
    LX = bin2Zmat(LX)
    SZ = bin2Zmat(SZ)

    ## find n - number of qubits
    n = np.max([binn(A) for A in [SX,LX,SZ]])
    SX = ZMatZeros((0,n)) if SX is None else ZMat(SX,n)

    ## Eq - all zeros vector (no signs for Z stabilisers in the default case)
    Eq = ZMatZeros((1,n))

    ## Input is SX, SZ
    if LX is None:
        ## convert SX to RREF mod 2
        SX,rowops = How(SX,2)
        ## SZ could be None - handle this case
        if SZ is None:
            SZ = ZMatZeros((0,n))
        else:
            ## otherwise, make RREF
            SZ = ZMat(SZ,n)
            SZ,rowops = How(SZ,2)
        ## Calculate LX from SX, SZ
        LX = CSSgetLX(SX,SZ)
    ## Input is SX, LX
    elif SZ is None:
        LX = ZMatZeros((0,n)) if LX is None else ZMat(LX,n)
        ## Calculate SZ from SX, LX
        SZ = getKer(np.vstack([SX,LX]),2)
    ## Find LZ
    if LZ is None:
        LZ = getKer(SX,2)
        ## canonical form of LX - LX LZ^T = I_k
        LZ,lx = LXZDiag(LZ,LX)
    return Eq,SX,LX,SZ,LZ

## for CSS code get LX
def CSSgetLX(SX, SZ):
    m,n = np.shape(SX)
    LiX = leadingIndices(SX)
    mask = 1 - set2Bin(n,LiX)
    SZ = SZ * mask
    SZ, rowops = How(SZ,2)
    LiZ = leadingIndices(SZ)
    LiE = sorted(set(range(n)).difference(set(LiX)).difference(set(LiZ)))
    k = len(LiE)
    nkr = len(LiZ)
    LX = ZMatZeros((k,n))
    for i in range(k):
        LX[i][LiE[i]] = 1
        for j in range(nkr):
            LX[i][LiZ[j]] = SZ[j][LiE[i]]
    LX, rowops = How(LX,2)
    return LX
	
## take logical X and Z
## arrange so that they are paired and have minimal overlap
def LXZCanonical(LX, LZ):
    LX, LZ = LXZDiag(LX,LZ)
    LZ, LX  = LXZDiag(LZ,LX)
    return LX, LZ

## helper for LXZCanonical
## updates LX
def LXZDiag(LX,LZ):
    A = matMul(LX, LZ.T,2)
    A, rowops = How(A,2)
    LX = doOperations(LX,rowops,2)[:len(A)]
    return LX, LZ    

#########################################
## Create CSS codes of various types
#########################################


## hypercube code of Kubica, Earl Campbell in r dimensions on 2^r qubits
def Hypercube(r):
    n = 1 << r
    SX = np.ones((1,n),dtype=int)
    LX = []
    a, b = 1, n >> 1
    for i in range(r):
        x = ([0] * a + [1]*a) * b
        LX.append(x)
        a = a << 1
        b = b >> 1
    return ZMat(SX), ZMat(LX)

## Reed Muller code on 2^r-1 qubits
def ReedMuller(r):
    SX = Mnt(r,r).T
    m,n = np.shape(SX)
    LX = np.ones((1,n),dtype=int)
    return SX,LX

## parse code from codetables.de
def CodeTable(mystr):
    SX, SZ, SXZ = [],[],[]
    mystr = mystr.replace(" ","")
    mystr = mystr.replace("[","")
    mystr = mystr.replace("]","")
    myarr = mystr.split('\n')
    for i in range(len(myarr)):
        x,z = myarr[i].split('|')
        x = str2ZMat(x)
        z = str2ZMat(z)
        if np.sum(x) == 0:
            SZ.append(z)
        else:
            SX.append(x)
            SXZ.append(z)
    return SX,SZ,SXZ


############################################################
## Symmetric Hypergraph Product Codes
############################################################


## Symmetric Hypergraph Product Code
def SHPC(T):
    T = bin2Zmat(T)
    H = matMul(T.T, T,2)
    return HPC(H,H)

## calculate HPC with canonical LX, LZ
## from Quintavalle et al Partitioning qubits in hypergraph product codes to implement logical gates
def HPC(A,B):
    A = bin2Zmat(A)
    B = bin2Zmat(B)
    ma,na = np.shape(A)
    mb,nb = np.shape(B)
    ## Generate SX
    C = np.kron(A,ZMatI(mb))
    D = np.kron(ZMatI(ma),B)
    SX = np.hstack([C,D])
    ## Generate SZ
    C = np.kron(ZMatI(na) ,B.T)
    D = np.kron(A.T,ZMatI(nb))
    SZ = np.hstack([C,D])

    ## calc LX
    Ha, Ka, Fa = Triag(A.T)
    Hb, Kb, Fb = Triag(B)    
    LX1 = np.kron(Fa,Kb).T
    LX1 = np.hstack([LX1,ZMatZeros((len(LX1),ma*mb))])
    LX2 = np.kron(Ka,Fb).T
    LX2 = np.hstack([ZMatZeros((len(LX2),na*nb)),LX2])
    LX = np.vstack([LX1,LX2])

    ## calc LZ
    Ha, Ka, Fa = Triag(A)
    Hb, Kb, Fb = Triag(B.T)
    LZ1 = np.kron(Ka,Fb).T
    LZ1 = np.hstack([LZ1,ZMatZeros((len(LZ1),ma*mb))])
    LZ2 = np.kron(Fa,Kb).T
    LZ2 = np.hstack([ZMatZeros((len(LZ2),na*nb)),LZ2])
    LZ = np.vstack([LZ1,LZ2])

    # pivots = [bin2Set(x * y)[0]  for x, y in zip(LX,LZ)]
    # print('pivots',pivots)
    # print('SHPC_partition(A,B)',SHPC_partition(A,B))

    return SX, SZ, LX, LZ

def SHCP_labels(m,n,s):
    return [(i,j,s) for i in range(m) for j in range(n)]

def SHPC_partition(T):
    T = bin2Zmat(T)
    H = matMul(T.T, T,2)
    m,n = np.shape(H)
    temp = SHCP_labels(n,n,'L')
    temp.extend( SHCP_labels(m,m,'R'))
    # print('labels',temp)
    lDict = dict()
    for i in range(len(temp)):
        a,b,s = temp[i]
        if a > b:
            a,b = b,a 
        l = (a,b,s)
        if l not in lDict:
            lDict[l] = []
        lDict[l].append(i)
    return list(lDict.values())

## Phase logical operator for SHPC and ZX symmetry codes
## if A is set, the qubits in A get S and the rest get S^3
def SHPC_operator(cList,A=None):
    S3,CZ = [], []
    for c in cList:
        if len(c) == 2:
            CZ.append(c)
        else:
            S3.append(c[0])
    ## if A not set, just set the first half of the singleton
    ## partition as S and the rest to S^3
    if A is None:
        A = S3[:len(S3) // 2]
    S3 = set(S3) - set(A)
    S3 = sorted(S3)
    S = [[s] for s in A]
    S3 = [[s] for s in S3]
    mystr =  f'S{S} S3{S3} CZ{CZ}'
    mystr = mystr.replace("], ", "]").replace("[[","[").replace("]]","]")
    return mystr

## generate H matrix which has symmetric weight LX and LZ
def genH(n,t=None):
    if t is None:
        t = n
    M = []
    R = list(range(2,t+1)) + [1]
    for k in R:
        M.extend([set2Bin(n,s) for s in iter.combinations(range(n),k)])
    return np.hstack([M,ZMatI(len(M))])

## repetition code
def repCode(r):
    SX = ZMatI(r-1)
    return np.hstack([SX,np.ones((r-1,1),dtype=int)])

## build 2D toric code from repetition code and SHPC constr
def toric2D(r):
    A = repCode(r)
    return SHPC(A)

## partition for toric code SS^3 logical operator
def toric2DPartition(r):
    A = repCode(r)
    return SHPC_partition(A)

## LxL Rotated Surface Code
def rotated_surface(L):
    n = L * L
    row_upper = [[[r*L + 2*c + (r % 2),r*L +  2*c + 1 + (r % 2)] for c in range(L // 2) ] for r in range(L)]
    row_lower = [[[r*L + 2*c + 1 - (r % 2), r*L + 2*c + 2 - (r % 2)] for c in range(L // 2) ] for r in range(L)]
    SX = []
    for i in range(L+1):
        if i == 0:
            SX.extend(row_upper[i])
        elif i == L:
            SX.extend(row_lower[i-1])
        else:
            rl = row_lower[i-1]
            ru = row_upper[i]
            SX.extend([rl[j] + ru[j] for j in range(len(rl))])
    LX = [r*L for r in range(L)]
    SX = [set2Bin(n,x) for x in SX]
    LX = [set2Bin(n,LX)]
    return SX,LX


###################################################
## Supporting Methods for Logical Operator Algs  ##
###################################################

## calculate q + u SX for wt(u) <= t
## iterator version
## if return_u, yield u 
def Orbit2distIter(Eq,SX,t=None,return_u=False):
    r, n = np.shape(SX)
    # r = len(SX)
    if t is None:
        t = r
    t = min(t, r)
    for q in Eq:
        for k in range(t+1):
            for xSupp in itertools.combinations(range(r),k):
                vSX = np.mod(q + np.sum(SX[xSupp,:],axis=0),2)
                if return_u:
                    u = set2Bin(r,xSupp)
                    yield vSX, u
                else:
                    yield vSX

## Same as Orbit2distIter, but return results in ZMat format
def Orbit2dist(Eq,SX,t=None,return_u=False):
    temp = list(Orbit2distIter(Eq,SX,t,return_u))
    if return_u:
        temp = list(zip(*temp))
        return [ZMat(a) for a in temp]
    else:
        return ZMat(temp)

#############################################################
## Code Analysis Tools
#############################################################

## generate codewords
def codewords(Eq,SX,LX):
    r,n = np.shape(SX)
    zvec = ZMatZeros((1,n))
    OSX = Orbit2dist(zvec, SX)

    LI, CW = [],[]
    for m,v in Orbit2distIter(Eq,LX,return_u=True):
        S = np.mod(m + OSX,2)
        LI.append(v)
        CW.append(S)
    return LI, CW

## print codewords
def state2str(S):
    return "+".join([ket(x) for x in S])

## Display |x> for states
def ket(x):
    return f'|{ZMat2str(x)}>'


def print_codewords(Eq,SX,LX):
    V, CW = codewords(Eq,SX,LX)
    print('\nCodewords')
    for i in range(len(V)):
        print(f'{ket(V[i])} : {state2str(CW[i])}')

def print_SXLX(SX,LX,SZ,LZ,compact=True):
    print('SX')
    for x in SX:
        print(x2Str(x) if compact else ZMat2str(x))
    print('LX')
    for x in LX:
        print(x2Str(x) if compact else ZMat2str(x))  
    print('SZ')
    for z in SZ:
        print(z2Str(z,2) if compact else ZMat2str(z))     
    print('LZ')
    for z in LZ:
        print(z2Str(z,2) if compact else ZMat2str(z)) 

## check if qVec is a logical operator by calculating [[X, CP_V(qVec)]] for each X in SX
## z, K_M are modulo 2N
def CPIsLO(qVec,SX,K_M,V,N,CP=True):
    for x in SX: 
        c = CPComm(qVec,x,V,N,CP)
        c, u = matResidual(K_M,c,2*N)
        c = np.mod(c,2*N) 
        if np.sum(c) != 0:
            print('x',x2Str(x))
            print('qVec',CP2Str(qVec,V,N))
            print('c',CP2Str(c,V,N))
            return False
    return True

## test if z is the Z-component of a logical XP operator
def isDiagLO(z,SX,K_M,N):
    for x in SX:
        ## componentwise product of x and z
        xz = x * z 
        ## Z and phase component of commutator 
        czp = np.mod(np.hstack([-2*xz,[np.sum(xz)]] ),N)
        ## residue of czp with respect to logical identities
        c, v = matResidual(K_M, czp, N)
        ## if non-zero, there is an error
        if np.sum(c) > 0:
            print(func_name(),f'Error: x={x},z={z},c={c}')
            return False 
    return True

## Logical actions of z in LZ
## LX are X-logicals
## z is modulo N
def DiagLOActions(LZ,LX,N):
    t = log2int(N)
    k,n = np.shape(LX)
    Eq = ZMatZeros((1,n))
    vLX,vList = Orbit2dist(Eq,LX,t,True)
    return ZMat(np.mod([[np.dot(z,x) for x in vLX] for z in LZ],N)),vList

# vList a list of logical indices
# pList a list of phases applied on each by an operator
# return a CP operator
def action2CP(vList,pVec,N):
    qVec = pVec.copy()
    for i in range(len(qVec)):
        v, p = vList[i],qVec[i]
        if p > 0:
            # ix = binInclusion(vList,v)
            ix = uLEV(v,vList)
            qVec = np.mod(qVec-p*ix,N)
            qVec[i] = p
    return qVec

def codeword_test(zCP,Eq,SX,LX,V,N):
    ## test action on each codeword is constant
    vList, CW = codewords(Eq,SX,LX)
    for i in range(len(CW)):
        s = {CPACT(e,zCP,V,N) for e in CW[i]}
        print(f'{ket(vList[i])}L {s}')
        # print(ket(vList[i]),"=",  state2str(CW[i]))

## action of CP operator zCP, V on LX
def action_test(zCP,Eq,LX,t,V):
    N = 1 << t
    ## test logical action
    Em,vList = Orbit2dist(Eq,LX,t,True)
    ## phases are modulo 2N 
    pList = ZMat([CPACT(e,zCP,V,N) for e in Em])//2

    # for i in range(len(Em)):
    #     e,v,p = Em[i],vList[i],pList[i]
    #     print(f'{ket(v)}L = {ket(e)} {p}')
    act = action2CP(vList,pList, N)
    # print('act',act)
    # mystr = CP2Str(2* act,vList, N)[1]
    # print("Action:", mystr)
    return 2*act, vList

######################################################
## Calculate X and Z distances
######################################################

def minWeight(SX):
    done = False
    w = np.sum(SX)
    SXt = [tuple(e) for e in SX]
    while not done:
        done = True
        for i in range(len(SX)):
            SXi,SXit = updateSX(SX,SXt,i)
            wi = np.sum(SXi)
            if wi < w:
                w = wi
                SX = SXi
                SXt = SXit
                done = False
    return SX
	
## add row i to all elements of SX
def updateSX(SX,SXt,i):
    SXw = np.sum(SX,axis=-1)
    e = SX[i]
    SXe = np.mod(SX+e,2)
    SXet = [tuple(e) for e in SXe]
    SXew = np.sum(SXe,axis=-1)

    # ix = SXew < SXw
    for j in range(len(SX)):
        if j != i and (SXew[j],SXet[j]) < (SXw[j],SXt[j]):
            SX[j] = SXe[j]
            SXt[j] = SXet[j]
    return SX,SXt

def ZDistance(SZ,LZ,LX):
    SZLZ = np.vstack([SZ,LZ])
    SZLZ, rowops = How(SZLZ,2)
    MW =  minWeight(SZLZ)
    LA = matMul(MW,LX.T,2)
    ix = np.sum(LA,axis=-1) > 0
    MW = MW[ix]
    LA = LA[ix]
    ix = np.argmin(np.sum(MW,axis=-1))
    return MW[ix], LA[ix]



########################################################
## Logical Identity Algorithm
########################################################

def LIAlgorithm(Eq,LX,SX,N,compact=True,debug=False):
    ## Logical identities
    KM = getKM(Eq, SX, LX, N)
    if debug:
        print(f'\nLogical Identities Precision N={N}:')
        print(f'K_M = Ker_{N}(E_M):')
        if compact:
            print(ZMat2compStr(KM))
        else:
            print(ZmatPrint(KM))
    return KM

########################################################
## Kernel Method
########################################################


## diagonal logical identity generators: Ker_N(E_M)
def getKM(Eq,SX, LX,N):
    t = log2int(N)
    A = Orbit2dist(Eq, np.vstack([SX,LX]), t)
    A = np.hstack([A,[[1]]*len(A)])
    return getKer(A,N)

## Algorithm 1: diagonal logical operators via Kernel method
## if target is not None - find LO of form CP_N(p,target)
def diagLOKer(Eq,LX,SX,N,target=None):
    r,n = np.shape(SX)
    ## t is orbit distance - if N = 2^t, use t else t is None
    t = log2int(N)
    ## Em are orbit reps 1-1 corresp with codewords
    ## only need to consider wt(v) <= t
    Em,vList = Orbit2dist(Eq,LX,t,True)
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
    E_L = Orbit2dist(Em,SX,t)
    ## calculate kernel modulo N
    K_L = getKer(E_L,N)
    return K_L, vList

# algorithm 1
def ker_method(Eq,LX,SX,N,compact=True):
    r,n = np.shape(SX)
    KL,V = diagLOKer(Eq,LX,SX,N)
    m = len(V)
    print(f'\nLogical Operators Precision N={N}:')
    print(f'K_L = Ker_{N}(E_L):')
    if compact:
        print(ZMat2compStr(KL))
    else:
        print(ZmatPrint(KL))

    print('\nLogical Operators and Phase Vectors:')
    print(f'z-component | p-vector')
    LZN, pList = KL[:,:n], np.mod(- KL[:,n:],N)
    KL, rowops = How(np.hstack([pList, LZN]),N)
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


## More flexible search - allows for operators of form S[0]S3[1]
def ker_search2(target,Eq,LX,SX,t,debug=False):
    r, n = np.shape(SX)
    k = len(LX)
    (x,qL), VL, t2 = Str2CP(target,n=k)
    if t > t2:
        qL = ZMat(qL) * (1 << (t-t2))
    N = 1 << t 
    SXLX = np.vstack([SX,LX])
    EL, uvList = Orbit2dist(Eq,SXLX,t,return_u = True)
    vList = uvList[:,r:]
    pList = np.mod(-ZMat([[CPACT(v,qL,VL,N)] for v in vList] )//2,N)
    KL = getKer(np.hstack([pList, EL]),N)
    # print(func_name(), 'KL')
    # print(ZmatPrint(KL))
    if KL[0,0] > 0:
        z = KL[0,1:]
        if debug:
            pList, V = DiagLOActions([z],LX,N)
            q = action2CP(V,pList[0],N)
            print('operator:',z2Str(z,N))
            print("action:",CP2Str(2*q,V,N)[1])
        return z
    if debug:
        print(func_name(),f'{target} Not found')
    return None

# def ker_search(target,Eq,LX,SX,N,compact=True):
#     ## convert target from string to binary vector
#     (x,z), V_L, t = Str2CP(target,n=len(LX))
#     j = bin2Set(z)
#     if len(j) != 1: 
#         print(f'\nCould not search for {target}')
#         return False
#     print(f'\nSearching for {target}')
#     binStr = V_L[j[0]]
#     KL,V = diagLOKer(Eq,LX,SX,N,binStr)
#     print(f'\nK_L = Ker_{N}(E_L):')
#     if compact:
#         print(ZMat2compStr(KL))
#     else:
#         print(ZmatPrint(KL))
#     ## swap phase component to first col to get action basis
#     r,n = np.shape(SX)
#     LZN, pList = KL[:,:n], np.mod(- KL[:,n:],N)
#     KL, rowops = How(np.hstack([pList, LZN]),N)
#     p, z = KL[0,0], KL[0,1:]
#     if p > 0:
#         print(f'\nFound Logical Operator:')
#         cpstr = CP2Str([2*p],[binStr],N)[1]
#         if compact:
#             print(cpstr,":", z2Str(z,N))
#         else:
#             print(cpstr,":", ZMat2str(z))
#     else:
#          print(f'\nLogical Operator {target} not found')


##########################################################
## Commutator Method
##########################################################

## return list of z components LZ and their action in terms of CP operators CPList
def CPLogicalOps(SX,LX,K_M, N):
    LZ = DiagLOComm(SX,K_M,N)
    LA, vList = DiagLOActions(LZ,LX,N)

    a, b = np.shape(LA)
    A = np.hstack([LA,LZ])
    A, rowops = How(A,N)
    LA, LZ = A[:,:b], A[:,b:]

    CPlist = ZMat([action2CP(vList,pList,N) for pList in LA])
    a, b = np.shape(CPlist)
    A = np.hstack([CPlist,LZ])
    A, rowops = How(A,N)
    CPlist, LZ = A[:,:b], A[:,b:]
    return LZ, vList, CPlist


## Algorithm 2: Commutator method for diagonal logical operators
## assume N is even

def DiagLOComm(SX,K_M,N):
    LZ = None
    for x in SX:
        Rx = commZ(x,K_M,N)
        LZ = Rx if LZ is None else nsIntersection([LZ, Rx],N)

    # print(func_name(), elapsedTime())
    return LZ

## span of commutators of x up to operators in K_M
def commZ(x,K_M,N):
    n = len(x)
    ## diag(x)
    xSet = bin2Set(x)
    ## rows of Ix are indicator vectors delta(x[i]==1)
    Ix = ZMat([set2Bin(n,[i]) for i in xSet])
    ## rows are of form (-2* delta(x[i]==1) | 1) because we require phase component equal to x.z 
    Rx = np.hstack([(N-2) * Ix,np.ones((len(Ix),1),dtype=int)])

    ## intersection with K_M
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

def comm_method(Eq, SX, LX, N,compact=True,debug=True):
    ## Logical identities
    K_M = LIAlgorithm(Eq,LX,SX,N,compact,debug=debug)
   
    zList = DiagLOComm(SX,K_M,N)
    pList, V = DiagLOActions(zList,LX,N)
    qList = ZMat([action2CP(V,pList,N) for pList in pList])
    ix = np.sum(qList,axis=-1) > 0 
    zList, pList, qList = zList[ix], pList[ix], qList[ix]
    if debug:
        print('\nApplying Commutator Method:')
        print('(action | z-component | q-vector)')
        for z, q in zip(zList,qList):
            if compact:
                print(z2Str(z,N),"|", row2compStr(q),"|", CP2Str(2*q,V,N)[1])
            else:
                print(ZMat2str(z), ZMat2str(q), CP2Str(2*q,V,N)[1])        

        print(f'\nq-vector Represents CP_{N}(2q, w) where w =')
        for i in range(len(V)):
            print(f'{i}: {ZMat2str(V[i])}')
    A = np.hstack([qList,zList])
    A, rowops = How(A,N)
    qList, zList = A[:,:len(V)], A[:,len(V):]
    ix = np.sum(qList,axis=-1) > 0 
    zList, qList = zList[ix], qList[ix]

    if debug:    
        print('\nRearranging matrix to form (q, z) and calculating Howell Matrix form:')
        print('(z-component | q-vector | action)')
        for z, q in zip(zList,qList):
            if compact:
                print(z2Str(z,N),"|", row2compStr(q),"|", CP2Str(2*q,V,N)[1])
            else:
                print(ZMat2str(z), ZMat2str(q), CP2Str(2*q,V,N)[1])     
    return zList, qList, V, K_M


############################################################
## Embedded Codes
############################################################

## canonical logical operators for any CP
def canonical_logical_algorithm(q1,V1,LZ,t,Vto=None,debug=True):
    # m,k = np.shape(V1)
    q1, V1 = CPNonZero(q1,V1)
    N = 1 << t
    # print('len(V1)',len(V1))
    ## q1, V1 are logical CP operators - convert to RP
    q2, V2 = CP2RP(q1,V1,t,CP=True,Vto=None)
    # print('len(V2)',len(V2))
    # print('q2',CP2Str(q2,V2,N,CP=False)[1])
    ## canonical logical in RP form
    q3 = q2 
    V3 = matMul(V2,LZ) 
    # print('len(V3)',len(V3))
    # print('q3',CP2Str(q3,V3,N,CP=False)[1])
    ## convert to CP - this will ensure max support size of operators is t
    q4,V4 = CP2RP(q3,V3,t,CP=False,Vto=Vto)
    # print('len(V4)',len(V4))

    q5,V5 = CP2RP(q4,V4,t,CP=True,Vto=Vto)
    # print('len(V5)',len(V5))
    if debug:
        print('Canonical Logical Operator Implementation - CP')
        print(CP2Str(q4,V4,N,CP=True)[1])
        print('Canonical Logical Operator Implementation - RP')
        print(CP2Str(q5,V5,N,CP=False)[1])
    return (q4,V4), (q5,V5)


############################################################
## Depth One Algorithm
############################################################

# search for transversal logical operator at level t of the Clifford hierarchy
def depth_one_t(Eq,SX,LX,SZ,LZ,t=2,debug=False):
    r,n = np.shape(SX)
    k,n = np.shape(LX)
    N = 1 << t

    ## All binary vectors of length n weight 1-t
    V = Mnt(n,t)
    
    ## Embedded Code
    ## move highest weight ops to left - more efficient in some cases
    V = np.flip(V,axis=0)
    SX_V = matMul(SX, V.T, 2)
    LX_V = matMul(LX, V.T, 2)
    Eq_V,SX_V,LX_V,SZ_V,LZ_V = CSSCode(SX_V,LX_V)
    SXLX = np.vstack([SX,LX])

    zList, qL, VL, K_M = comm_method(Eq_V, SX_V, LX_V, N,compact=True,debug=False)
    qRP = zList[0] * 2
    zList = zList[1:]
    PRKM = np.hstack([zList * 2, ZMatZeros((len(zList),1))])
    PRKM = np.vstack([2* K_M, PRKM])
    PRKM, rowops = How(PRKM, 2*N)
    qCP = CP2RP(qRP,V,t,CP=False,Vto=V)[0]

    ## add all zeros row to end of V, qCP and qPR representing phase component
    V = np.vstack([V,ZMatZeros((1,n))])
    qCP = np.hstack([qCP,[0]])
    qRP = np.hstack([qRP,[0]])
    # PRKM = np.hstack([PRKM,ZMatZeros((len(PRKM),1))])

    CPKM = ZMat([CP2RP(q,V,t,CP=False,Vto=V)[0] for q in PRKM])
    ## CPKM is modulo  2N
    CPKM, rowops = How(CPKM,2*N)

    if debug:
        print('V')
        for i in range(len(V)):
            print(i,'=>',bin2Set(V[i]))
        
        print('Embedded CSS Code')
        print_SXLX(SX_V,LX_V,SZ_V,LZ_V,True)

        print('Target - RP Form',ZMat2str(qRP))
        print(CP2Str(qRP,V,N,False)[1])
        print('Target - CP Form',ZMat2str(qCP) )
        print(CP2Str(qCP,V,N,True)[1])

        print("\nLogical Identites + Operators - RP Form")
        print(ZmatPrint(PRKM))

        print("\nLogical Identites + Operators - CP Form")
        print(ZmatPrint(CPKM))


        pList, VL = DiagLOActions([qRP[:-1]//2],LX_V,N)
        qL = action2CP(VL,pList[0],N)
        target = "".join(CP2Str(2*qL,VL,N))
        print('Logical action:',target)

    sol = findDepth1(CPKM,qCP,V,N)

    if sol is not False: 
        print("\nDepth-One Solution Found")
        qCP =  sol
        cList = CP2Partition(qCP,V)
        print('Qubit Partition', cList)
        op = CP2Str(qCP, V, N)[1]
        print('Logical Operator:', op)
        
        qRP = CP2RP(qCP,V,t,CP=True,Vto=V)[0]
        qRP = qRP[:-1]//2

        pList, VL = DiagLOActions([qRP],LX_V,N)
        qL = action2CP(VL,pList[0],N)
        target = "".join(CP2Str(2*qL,VL,N))
        print('Logical action:',target)
        if len(LX) < 10:
            print('Testing Action on Codewords')
            codeword_test(qCP,Eq,SX,LX,V,N)
        print('LO Test:',CPIsLO(qCP,SX,CPKM,V,N))
        return cList, target
    else:
        print("\nNo Depth-One Solution Found")

def depth_one_algorithm(Eq,SX,LX,SZ,LZ,target,cList=None,debug=False):
    r,n = np.shape(SX)
    k,n = np.shape(LX)
    (x,qL), VL, t = Str2CP(target,n=k)
    N = 1 << t
    if cList is None:
        ## All binary vectors of length n weight 1-t
        V = Mnt(n,t)
    else:
        V = Mnt_partition(cList,n,t)

    ## Embedded Code
    ## move highest weight ops to left - more efficient in some cases
    V = np.flip(V,axis=0)
    SX_V = matMul(SX, V.T, 2)
    LX_V = matMul(LX, V.T, 2)
    Eq_V,SX_V,LX_V,SZ_V,LZ_V = CSSCode(SX_V,LX_V)
    SXLX = np.vstack([SX,LX])

    if cList is None:
        ## Canonical form of LO
        (qCP,V1),(qPR,VL2) = canonical_logical_algorithm(qL,VL,LZ,t,V)
    else:
        print(f'\nKernel Method - Search for {target}')
        qPR = ker_search2(target,Eq_V,LX_V,SX_V,t,debug=True)
        if qPR is None: 
            print(f'Could not find {target} with this partition')
            return None
        qPR = 2 * qPR
        qCP,V2 = CP2RP(qPR,V,t,False,V)

    ## Logial Identities - KM modulo 2N
    PRKM = getKM(Eq_V,SX_V,LX_V,N) * 2

    ## add all zeros row to end of V, qCP and qPR representing phase component
    V = np.vstack([V,ZMatZeros((1,n))])
    qCP = np.hstack([qCP,[0]])
    qPR = np.hstack([qPR,[0]])

    CPKM = ZMat([CP2RP(q,V,t,CP=False,Vto=V)[0] for q in PRKM])
    ## CPKM is modulo N - change to mod 2N
    CPKM, rowops = How(CPKM,2*N)

    if debug:
        print('V')
        for i in range(len(V)):
            print(i,'=>',bin2Set(V[i]))
        
        print('Embedded CSS Code')
        print_SXLX(SX_V,LX_V,SZ_V,LZ_V,True)

        print('Target - RP Form',ZMat2str(qPR))
        print(CP2Str(qPR,V,N,False)[1])
        print('Target - CP Form',ZMat2str(qCP) )
        print(CP2Str(qCP,V,N,True)[1])

        # print("\nLogical Identites - RP Form")
        # print(ZmatPrint(PRKM))

        # print("\nLogical Identites - CP Form")
        # print(ZmatPrint(CPKM))

        print('CPIsLO',CP2Str(qPR,V,N,False)[1],CPIsLO(qPR,SX,PRKM,V,N,CP=False))               
        print('LI test for PRKM')
        for q in PRKM:
            print('Testing',CP2Str(q,V,N, False)[1],CPIsLO(q,SXLX,PRKM,V,N, False))

        print('LI test for CPKM')
        for q in CPKM:
            print('Testing',CP2Str(q,V,N)[1],CPIsLO(q,SXLX,CPKM,V,N))

        if k < 10:
            print('Logical Operator Test')
            print('Testing',CP2Str(qCP,V,N)[1],CPIsLO(qCP,SX,CPKM,V,N))
            codeword_test(qCP,Eq,SX,LX,V,N)

    sol = findDepth1(CPKM,qCP,V,N)
    # print(sol)
    if sol is not False: 
        print("\nDepth-One Solution Found")
        qCP =  sol
        cList = CP2Partition(qCP,V)
        print('Qubit Partition', cList)
        
        print('Logical Operator:', CP2Str(qCP, V, N)[1])
        print('LO Test:',CPIsLO(qCP,SX,CPKM,V,N))
        if len(LX) < 10:
            print('Testing Action on Codewords')
            codeword_test(qCP,Eq,SX,LX,V,N)
        return cList
    else:
        print("\nNo Depth-One Solution Found")


## take LR configuration
## LR[i] is the list of indices j with LR[j] = i
def LR2ix(LR):
    ix = [[] for i in range(3)]
    for i in range(len(LR)):
        ix[LR[i]].append(i)
    return ix

def CPACT(e,z,V,N):   
    xrow = np.abs(uGEV(e,V))
    # CP_scale = 1 << np.sum(V,axis=-1)
    return np.sum(xrow * z ) % (2*N)
    
## return indices to restore original col order
def ixRev(ix):
    ixR = [0] * len(ix)
    for i in range(len(ix)):
        ixR[ix[i]] = i 
    return ixR

## return list of col indices which overlap with Erm[m]
def FDoverlap(j, V):
    ## multiply Erm by Erm[m] and sum cols
    ## yields number of places supp of col overlaps with supp of Erm[m]
    E = np.sum(V[j] * V,axis=1)
    return [i for i in range(len(V)) if E[i] > 0 and i != j]

## Given an embedded code, find depth 1 logical operator
def findDepth1(KM,qVec,V,N):
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
        qVec1 = HowRes(KM[:,ix], qVec[ix],2*N)
        # print('qVec1',qVec1)
        ## s is the support of z1
        s = bin2Set(qVec1)
        m = min(s)
        ## if m < LR[0] then it was not eliminated - invalid configuration
        if m >= len(LRix[0]): 
            ## indices corresp to original col order
            ixR = ixRev(ix)

            ## reorder z1
            qVec1 = qVec1[ixR] 
            ## in this case, it's a valid solution as only non-zero values are on RHS
            if m >= len(LRix[0]) + len(LRix[1]): 
                return qVec1
            ## original order of m
            m1 = ix[m] 
            ## get overlap with m
            ix = FDoverlap(m1, V) 
            ## LR1: overlap with m moved to left; m moved to the right
            LR1 = list(LR) 
            for i in ix:
                LR1[i] = 0    
            LR1[m1] = 2
            ## LR: m moved to left
            LR2 = list(LR)
            LR2[m1] = 0
            for A in LR2, LR1:
                A = tuple(A)
                ## add to visited and todo if not already encountered
                if A not in visited:
                    visited.add(A) 
                    todo.append(A)
    ## we have looked at all possible configurations with no valid solution
    return False

############################################################
## Generate Codes with Desired Logical Operators
############################################################

## kron of a list of matrices
def kronList(AList):
    temp = AList[0]
    for i in range(1,len(AList)):
        temp = np.kron(temp,AList[i])
    return temp

## Logical Z for k dimensional toric code of distance d
def zMatrix(k,d):
    Ik = ZMatI(k)
    rd = ZMat([0] * (d-1) + [1])
    ad = ZMat([1] * d)
    temp = []
    for i in range(k):
        kList = [Ik[i]] + [rd] * k
        kList[i+1] = ad
        temp.append(kronList(kList))
    return np.vstack(temp)

## search for a code with logical operator target using LZ of weight d
def codeSearch(target, d, debug=False):
    (x,q1), V1, t = Str2CP(target)
    N = 1 << t
    k = len(x)
    n = k * d
    compact = n > 15
    ## calculate logical Z operators
    LZ = zMatrix(k,d)
    ## get canonical implementation of target
    (q2,V2),(q1,V1) = canonical_logical_algorithm(q1,V1,LZ,t,debug=False,Vto=None)
    ## SXLX are non-zero cols rows of V1
    SXLX = RemoveZeroRows(V1.T)
    ## Split into SX and LX
    SX, LX = [], []
    for i in range(k):
        ## get logical X
        x = SXLX[d*i + d-1]
        LX.append(x)
        ## add logical X to SX rows in the block
        for j in range(d-1):
            y = SXLX[d*i + j]
            SX.append(np.mod(x+y,2))
    ## make a CSS code
    Eq,SX,LX,SZ,LZ = CSSCode(SX,LX)

    r, n = np.shape(SX)
    compact = n > 15

    if debug:
        print('Embedded Code Checks and Logicals')
        print_SXLX(SX,LX,SZ,LZ,compact)

        ## Logical identities
        K_M = getKM(Eq, SX, LX, N)

        ## Algorithm 1 - search
        print(f'\nKernel Method - Search for {target}')
        ker_search2(target,Eq,LX,SX,N,compact)

        print('Qubits in Code:',n)
        ## Z distance
        z,pVec = ZDistance(SZ,LZ,LX)
        print('Z-Distance',np.sum(z))
        print(f'Z{bin2Set(z)} = Logical Z{bin2Set(pVec)}')

        ## X distance
        z,pVec = ZDistance(SX,LX,LZ)
        print('X-Distance',np.sum(z))
        print(f'X{bin2Set(z)} = Logical X{bin2Set(pVec)}')

    else:
        z,pVec = ZDistance(SZ,LZ,LX)
        x,pVec = ZDistance(SX,LX,LZ)
        print(f'{d} {n} {np.sum(x)} {np.sum(z)}')
