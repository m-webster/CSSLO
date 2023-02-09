from common import *
import itertools as iter
import json
import os
from NSpace import *
from XCP_algebra import *


##########################################
## CSS Codes
##########################################

def CSSCode(SX,LX=None,SZ=None,LZ=None):
    """Create CSS code from various input types.

    Keyword arguments:
    SX -- X-checks 
    LX -- X-logicals (optional)
    SZ -- Z-checks (optional)
    LZ -- Z-logicals (optional)

    Default is to give values for SX and LX, then calculate SZ and LZ.
    Can take either text or arrays as input.
    Returns Eq,SX,LX,SZ,LZ. Eq is a zero vector of shape (1,n).
    """
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
    """Get LX for CSS code with check matrices SX, SZ."""
    m,n = np.shape(SX)
    LiX = [leadingIndex(x) for x in SX]
    mask = 1 - set2Bin(n,LiX)
    SZ = SZ * mask
    SZ, rowops = How(SZ,2)
    LiZ = [leadingIndex(x) for x in SZ]
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
    """Modify LX, LZ such that LX LZ^T = I if possible."""
    LX, LZ = LXZDiag(LX,LZ)
    LZ, LX  = LXZDiag(LZ,LX)
    return LX, LZ

## helper for LXZCanonical
## updates LX
def LXZDiag(LX,LZ):
    """Convert LX to a form with minimal overlap with LZ."""
    A = matMul(LX, LZ.T,2)
    A, rowops = How(A,2)
    LX = doOperations(LX,rowops,2)[:len(A)]
    return LX, LZ    

#########################################
## Create CSS codes of various types
#########################################


## hypercube code of Kubica, Earl Campbell in r dimensions on 2^r qubits
def Hypercube(r):
    """Create Hypercube code in r dimension on 2^r qubits.
    Returns SX, LX.
    """
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
    """Create Reed Muller code on 2^r-1 qubits
    Returns SX, LX.
    """
    SX = Mnt(r,r).T
    m,n = np.shape(SX)
    LX = np.ones((1,n),dtype=int)
    return SX,LX

# ## Reed Muller code on 2^r-1 qubits
def ReedMullerMod(r,t):
    ## odd values of t
    R = range(1,2*t,2)
    ## drop first row of Mnt
    SX = [set2Bin(r,s) for k in R for s in itertools.combinations(range(r),k)]
    SX = ZMat(SX[1:]).T
    ## LX is first row, remainder is SX
    LX = ZMat([SX[0]])
    SX = SX[1:]
    return SX,LX


## parse code from codetables.de
def CodeTable(mystr):
    """parse check matrix from codetables.de.
    Returns SX, SZ and SXZ (Z components of X-checks)
    """
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
## Triorthogonal Codes from Classification of Small Triorthogonal Codes
############################################################

def triCodeData():
    '''Extract triorthogonal code data from file
    Returns: list of polynomials defining the triorthogonal codes.
    '''
    mypath = sys.path[0] + "/"
    myfile = 'triorthogonal.txt'
    f = open(mypath + myfile, "r")
    mytext = f.read()
    mytext = mytext.split("\n")
    temp = []
    for r in mytext:
        temp.append(r.split("\t"))
    return temp

def TriCodeParams(triRow):
    '''Display key row data from triRow as a string.'''
    return f'ix={triRow[0]}; r=max(k)={triRow[2]}; c=max(n)={triRow[3]}; f(x)={triRow[1]}'

def parseTriCode(triRow):
    '''Parse the data from triRow and return a representation of the polynomial in list form.'''
    r = int(triRow[2])
    A = ZMat([set2Bin(r,p) for p in parseTriPoly(triRow[1])])
    return A[:,1:]

def getTriCode(triRow,k):
    '''Return check matrix for triRow with k logical qubits.'''
    A = parseTriCode(triRow)
    m,r = np.shape(A)
    temp = []
    V = np.vstack([ZMatZeros((1,r)), Mnt(r,r)])
    if np.sum(A) == 0:
        c = len(V)
        G = V.T 
    else:
        myval = []
        for v in A:
            myval.append(uLEV(v,V))
        ix = np.mod(np.sum(myval,axis=0),2)
        c = np.sum(ix)
        G = V[ix==1].T
    G = np.vstack([[1]*c,G])
    SX, rowops = How(G,2)
    if len(SX) < k:
        return None
    SX = SX[:,k:]
    LX = SX[:k]
    SX = SX[k:]
    return SX,LX

def parseTriPoly(poly):
    '''Parse a triorthogonal polynomial in text form from file.'''
    tokens = []
    t = ''
    for c in poly:
        if c in ["(",")","+"]:
            if len(t) > 0:
                tokens.append(t)
            t = ''
            tokens.append(c)
        else:
            t += c
    if len(t) > 0:
        tokens.append(t)
    inPar = 0
    temp = []
    for t in tokens:
        if t == "(":
            inPar = 1
            brack = []
        if t == ")":
            inPar = 2
        if t not in ["(",")","+"]:
            if t[0] == "x":
                myterm = t[1:].split('x')
                myterm = [int(c) for c in myterm]
            else:
                myterm = []
            if inPar == 0:
                temp.append(myterm)
            if inPar == 1:
                brack.append(myterm)
            if inPar == 2:
                for b in brack:
                    temp.append(b + myterm)
                inPar = 0
    return temp

def tOrthogonal(SX):
    '''Return largest t for which the weight of the product of any t rows of SX is even.'''
    t = 0
    r, n = np.shape(SX)
    while t < r:
        ## all combinations of t+1 rows
        for ix in iter.combinations(range(r),t+1):
            T = np.prod(SX[ix,:],axis=0)
            if np.sum(T) % 2 == 1:
                return t 
        t += 1
    return t

def nkdReport(SX,LX,SZ,LZ):
    '''Report [[n,k,dX,dZ]] for CSS code specified by SX,LX,SZ,LZ.'''
    k,n = np.shape(LX)
    Zop, Zact = ZDistance(SZ,LZ,LX)
    Xop, Xact = ZDistance(SX,LX,LZ)
    return(f'n:{n} k: {k} dX: {sum(Xop)} dZ:{sum(Zop)}')


############################################################
## 2D Hyperbolic Tesselations
############################################################


def importCodeList(myfile):
    """Import hyperbolic surface codes stored in myfile.
    Records in myfile are stored in JSON format.
    Parse each record and return list of dict codeList.
    """
    mypath = sys.path[0] + "/hyperbolic_codes/"
    f = open(mypath + myfile, "r")
    mytext = f.read()
    mytext = mytext.replace("\\\n","").replace("\n","")
    mytext = mytext.replace("}","}\n")
    mytext = mytext.split("\n")
    codeList = []
    for myline in mytext:
        if len(myline) > 0:
            myrow = json.loads(myline)
            codeList.append(myrow)
    f.close()  
    return codeList

def printCodeList(codeList,myfile):
    '''Print parameters of the hyperbolic codes stored in list of dict codeList'''
    temp = []
    temp.append(f'Codes in File {myfile}:\n')
    temp.append(f'i\tindex\tV\tE\tF')
    for i in range(len(codeList)):
        myrow = codeList[i]
        V = myrow["zEV"].count("\n")+1
        F = myrow["zEF"].count("\n")+1
        E = myrow["zEV"].index("\n") if  myrow["zEV"].count("\n") > 0 else len(myrow["zEV"])
        ix = myrow["index"]
        temp.append(f'{i}\t{ix}\t{V}\t{E}\t{F}')
    return "\n".join(temp)

def hypColour(myrow):
    '''Make a hyperbolic colour code based on Face-Vertex adjacency matrix stored in myrow.'''
    SX = str2ZMatdelim( myrow['zFV'])
    return SX, SX

def hypCode(myrow):
    '''Make a hyperbolic surface code based on Face-Edge and Vertex-Edge adjacency matrix stored in myrow.'''
    SX = str2ZMatdelim( myrow['zEV'])
    SZ = str2ZMatdelim( myrow['zEF'])
    return SX, SZ


############################################################
## Symmetric Hypergraph Product Codes
############################################################


## Symmetric Hypergraph Product Code
def SHPC(T):
    '''Make symmetric hypergraph product code from T.
    T can either be a string or np array.
    Returns SX, SZ.'''
    T = bin2Zmat(T)
    H = matMul(T.T, T,2)
    return HPC(H,H)

def HPC(A,B):
    '''Make hypergraph product code from clasical codes A, B
    A and B can either be a string or np array.
    Returns SX, SZ.'''
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
    return SX, SZ

def SHCP_labels(m,n,s):
    '''Return labels of symmetric hypergraph product code.
    Assumes m x n grid and s is either R or L.'''
    return [(i,j,s) for i in range(m) for j in range(n)]

def SHPC_partition(T):
    '''Calculate qubit partition resulting in a depth-one transversal gate for an SHPC code.'''
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
    '''Calculate partitioned logical operator for an SHPC code based on qubit partition cList.
    If A is set, this indicates which of the fixed qubits have an S operator applied to them. '''
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
    '''Generate matrix suitable for producing a SHPC code.'''
    if t is None:
        t = n
    M = []
    R = list(range(2,t+1)) + [1]
    for k in R:
        M.extend([set2Bin(n,s) for s in iter.combinations(range(n),k)])
    return np.hstack([M,ZMatI(len(M))])

## repetition code
def repCode(r,closed=True):
    '''Generate classical repetition code on r bits.
    If closed, include one dependent row closing the loop.'''
    s = r if closed else r-1 
    SX = ZMatZeros((s,r))
    for i in range(s):
        SX[i,i] = 1
        SX[i,(i+1)%r] = 1
    return SX
    SX = ZMatI(r-1)
    zr = ZMatZeros((r-1,1))
    return np.hstack([SX,zr]) + np.hstack([zr,SX]) 
    return np.hstack([SX,np.ones((r-1,1),dtype=int)])

## build 2D toric code from repetition code and SHPC constr
def toric2D(r):
    '''Generate distance r 2D toric code using SHCP construction.
    Returns SX, SZ.'''
    A = repCode(r)
    return SHPC(A)

## partition for toric code SS^3 logical operator
def toric2DPartition(r):
    '''Generate qubit partition for distance r 2D toric code.'''
    A = repCode(r)
    return SHPC_partition(A)

## LxL Rotated Surface Code
def rotated_surface(L):
    '''Generate distance L rotated surface code.
    Returns SX, LX.'''
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


#########################################################################
## Poset Codes from Self-Orthogonal Codes Constructed from Posets and Their Applications in Quantum Communication
#########################################################################

def subsets(r):
    '''Generate list of subsets of [0..r]'''
    temp = []
    for s in range(len(r)+1):
        temp.extend(list(iter.combinations(r,s)))
    return temp

def subIdeals(a,b):
    '''Return subideal of [0..a-1],[0..b-1] in a poset.
    if b==0, this is just the subsets of [a..a-1].
    if b>0, these are of form {0..a-1} u s where s is a subset of [0..b-1].
    '''
    if b == 0:
        return subsets(range(a))
    aSet = list(range(a))
    return [tuple(aSet + list(bSet)) for bSet in subsets(range(a,b+a))]

def posetCode(a1,a2,b1,b2):
    '''Return Poset code where a1>=a2,b1>=b2 are integers >= 0.
    columns of SX are binary vectors of length a1 + b1 corresponding to the subIdeals of (a1,b1) which are not subIdeals of (a2,b2).
    '''
    if a2 > a1 or b2 > b1:
        return False
    I1 = subIdeals(a1,b1)
    I2 = set(subIdeals(a2,b2))
    SX = []
    for r in I1:
        if r not in I2:
            SX.append(set2Bin(a1+b1,r))
    return ZMat(SX).T

###################################################
## Supporting Methods for Logical Operator Algs  ##
###################################################


def Orbit2distIter(Eq,SX,t=None,return_u=False):
    '''Interator yielding binary rows of form (q + u SX \mod 2) for q in Eq and wt(u) <= t.
    if return_u, yield u as well as the row.'''
    r, n = np.shape(SX)
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

def Orbit2dist(Eq,SX,t=None,return_u=False):
    '''Matrix with binary rows of form (q + u SX \mod 2) for q in Eq and wt(u) <= t.
    if return_u, yield u as well as the row.'''
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
    '''Return canonical codewords LI={v}, CW={sum_u (q + uSX + vLX)}'''
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
    '''Print a state corresponding to a binary matrix S in the form \sum_{x \in S}|x>'''
    return "+".join([ket(x) for x in S])

## Display |x> for states
def ket(x):
    '''Display |x> for states.'''
    return f'|{ZMat2str(x)}>'


def print_codewords(Eq,SX,LX):
    '''Print the canonical codwords of a CSS code defined by X-checks SX and X-logicals LX'''
    V, CW = codewords(Eq,SX,LX)
    print('\nCodewords')
    for i in range(len(V)):
        print(f'{ket(V[i])} : {state2str(CW[i])}')

def print_SXLX(SX,LX,SZ,LZ,compact=True):
    '''Print the X-checks, X-logicals, Z-checks, Z-logicals of a CSS code.
    If compact=True, print full vector representations, otherwise print support of the vectors.'''
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


def CPIsLO(qVec,SX,K_M,V,N,CP=True):
    '''Check if qVec is a logical operator by calculating the group commutator [[A, CP_V(qVec)]] for each A in SX
    Inputs: 
    qVec: Z_2N vector of length |V| representing a product of CP operators \prod_{v \in V}CP_N(qVec[v],v)
    SX: binary matrix representing X-checks
    K_M: Z_2N matrix representing phase and z-components of logical identities K_M = ker_{2N}(E_M)
    V: binary matrix representing vector part of CP operators
    N: precision of operators
    CP: if True, treat operators as CP operators, otherwise RP operators'''
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
        c, v = matResidual(K_M, czp, N)
        ## if non-zero, there is an error
        if np.sum(c) > 0:
            print(func_name(),f'Error: x={x},z={z},c={c}')
            return False 
    return True


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
    k,n = np.shape(LX)
    Eq = ZMatZeros((1,n))
    vLX,vList = Orbit2dist(Eq,LX,t,True)
    pVec = ZMat(np.mod([[np.dot(z,x) for x in vLX] for z in LZ],N))
    return pVec,vList


def action2CP(vList,pVec,N):
    '''Return controlled phase operator corresponding to phase vector pVec
    pList a list of phases applied on each by an operator
    return a CP operator'''
    qVec = pVec.copy()
    for i in range(len(qVec)):
        v, p = vList[i],qVec[i]
        if p > 0:
            # ix = binInclusion(vList,v)
            ix = uLEV(v,vList)
            qVec = np.mod(qVec-p*ix,N)
            qVec[i] = p
    return qVec

def codeword_test(qVec,Eq,SX,LX,V,N):
    '''Print report on action of \prod_{v \in V}CP_N(qVec[v],v) operator on comp basis elts in each codeword.
    Inputs:
    qVec: Z_2N vector of length |V| representing CP operator
    Eq:
    SX: X-checks in binary matrix form
    LX: X-logicals in binary matrix form
    V: vectors for CP operator
    N: precision of CP operator'''
    vList, CW = codewords(Eq,SX,LX)
    for i in range(len(CW)):
        s = {CPACT(e,qVec,V,N) for e in CW[i]}
        print(f'{ket(vList[i])}L {s}')

def action_test(qVec,Eq,LX,t,V):
    '''Return action of \prod_{v \in V}CP_N(qVec[v],v) operator on codewords in terms of CP operators.'''
    N = 1 << t
    ## test logical action
    Em,vList = Orbit2dist(Eq,LX,t,True)
    ## phases are modulo 2N 
    pList = ZMat([CPACT(e,qVec,V,N) for e in Em])//2
    act = action2CP(vList,pList, N)
    return 2*act, vList

######################################################
## Calculate X and Z distances
######################################################

def minWeight(SX):
    '''Return a set of vectors spanning <SX> which have minimum weight.'''
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
	

def updateSX(SX,SXt,i):
    '''Add row i to all elements of SX - helper function for minWeight'''
    SXw = np.sum(SX,axis=-1)
    e = SX[i]
    SXe = np.mod(SX+e,2)
    SXet = [tuple(e) for e in SXe]
    SXew = np.sum(SXe,axis=-1)
    for j in range(len(SX)):
        if j != i and (SXew[j],SXet[j]) < (SXw[j],SXt[j]):
            SX[j] = SXe[j]
            SXt[j] = SXet[j]
    return SX,SXt

def ZDistance(SZ,LZ,LX):
    '''Find lowest weight element of <SZ,LZ> which has a non-trivial logical action.
    Return z component and action'''
    SZLZ = np.vstack([SZ,LZ])
    SZLZ, rowops = How(SZLZ,2)
    MW =  minWeight(SZLZ)
    LA = matMul(MW,LX.T,2)
    ix = np.sum(LA,axis=-1) > 0
    ## cover the case where there are no nontrivial LO - min weight of stabilisers
    if np.any(ix):
        MW = MW[ix]
        LA = LA[ix]
    ix = np.argmin(np.sum(MW,axis=-1))
    return MW[ix], LA[ix]

########################################################
## Logical Identity Algorithm
########################################################

def LIAlgorithm(Eq,LX,SX,N,compact=True,debug=False):
    '''Run logical Identity Algorithm.
    Inputs:
    Eq: 1xn zero vector
    LX: X logicals
    SX: X-checks
    N: required precision
    compact: print full vector form if True, else support form
    debug: verbose output
    Output:
    KM: Z_N matrix representing phase and z components of diagonal logical identities.'''
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



def getKM(Eq,SX, LX,N):
    '''Return KM - Z_N matrix representing phase and z components of diagonal logical identities.'''
    t = log2int(N)
    A = Orbit2dist(Eq, np.vstack([SX,LX]), t)
    A = np.hstack([A,[[1]]*len(A)])
    return getKer(A,N)


def diagLOKer(Eq,LX,SX,N,target=None):
    '''Return diagonal logical operators via Kernel method.
    Inputs:
    Eq: 1xn zero vector
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

def ker_method(Eq,LX,SX,N,compact=True):
    '''Run the kernel method and print results.
    Inputs:
    Eq: 1xn zero vector
    LX: X logicals
    SX: X-checks
    N: required precision
    compact: if True, output full vector forms, otherwise support view.
    '''
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



def ker_search(target,Eq,LX,SX,t,debug=False):
    '''Run kernel search algorithm.
    Inputs:
    target: string corresponding logical CP operator to search for
    Eq: 1xn zero vector
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
    ## check if top lhs is 1 - in this case, the LO has been found
    if KL[0,0] == 1:
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

##########################################################
## Commutator Method
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
    A, rowops = How(A,N)
    LA, LZ = A[:,:b], A[:,b:]

    CPlist = ZMat([action2CP(vList,pList,N) for pList in LA])
    a, b = np.shape(CPlist)
    A = np.hstack([CPlist,LZ])
    A, rowops = How(A,N)
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
    '''Run the commutator method and print results.
    Inputs:
    Eq: 1xn zero vector
    LX: X logicals
    SX: X-checks
    N: required precision
    compact: if True, output full vector forms, otherwise support view.
    debug: if True, verbose output.
    Output:
    zList: list of z-components generating non-trivial diagonal XP operators
    qList: list of q-vectors corresponding to logical action of each operator
    V: vectors indexing qList
    K_M: phase and z components of diagonal logical XP identities 
    '''
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


def canonical_logical_algorithm(q1,V1,LZ,t,Vto=None,debug=True):
    '''Run canonical logical operator algorithm for given code and target CP operator
    Inputs:
    q1: q-vector for CP operator \prod_{v in V1}CP_N(q[v],v)
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
    V3 = matMul(V2,LZ) 
    ## convert to CP - this will ensure max support size of operators is t
    q4,V4 = CP2RP(q3,V3,t,CP=False,Vto=Vto)

    q5,V5 = CP2RP(q4,V4,t,CP=True,Vto=Vto)
    if debug:
        print('Canonical Logical Operator Implementation - CP')
        print(CP2Str(q4,V4,N,CP=True)[1])
        print('Canonical Logical Operator Implementation - RP')
        print(CP2Str(q5,V5,N,CP=False)[1])
    return (q4,V4), (q5,V5)


############################################################
## Depth One Algorithm
############################################################


def depth_one_t(Eq,SX,LX,t=2,cList=None, debug=False):
    '''Run depth-one algorithm - search for transversal logical operator at level t of the Clifford hierarchy
    Inputs:
    Eq: 1xn zero vector
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
    
    ## Embedded Code
    ## move highest weight ops to left - more efficient in some cases
    V = np.flip(V,axis=0)
    SX_V = matMul(SX, V.T, 2)
    LX_V = matMul(LX, V.T, 2)
    Eq_V,SX_V,LX_V,SZ_V,LZ_V = CSSCode(SX_V,LX_V)
    SXLX = np.vstack([SX,LX])

    zList, qL, VL, K_M = comm_method(Eq_V, SX_V, LX_V, N,compact=True,debug=False)
    tList = [CPlevel(2*q,VL,N) for q in qL]
    ## Level t logical operators
    ix = [i for i in range(len(tList)) if tList[i] == t]
    if len(ix) == 0:
        print(f'No level {t} logical operators found.')
        return None
    j = min(ix)
    zList = [a for a in zList]
    qRP = zList.pop(j) * 2
    zList = ZMat(zList,len(qRP))
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
    

def ixRev(ix):
    ## return indices to restore original column order
    ## input: ix - permutation of [0..n-1]
    ## output: ixR such that ix[ixR] = [0..n-1]
    ixR = [0] * len(ix)
    for i in range(len(ix)):
        ixR[ix[i]] = i 
    return ixR

def FDoverlap(j, V):
    '''return list of indices i in [0..|V|-1] not equal to j such that wt(V[i]V[j]) > 0'''
    E = np.sum(V[j] * V,axis=1)
    return [i for i in range(len(V)) if E[i] > 0 and i != j]


def findDepth1(KM,qVec,V,N):
    '''Run Depth-one algorithm. 
    Inputs:
    KM: Z_2N matrix representing logical identities and operators of the embedded code
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


def kronList(AList):
    '''Return kronecker product of a list of matrices AList'''
    temp = AList[0]
    for i in range(1,len(AList)):
        temp = np.kron(temp,AList[i])
    return temp


def zMatrix(k,d):
    '''Return Z-Logicals for k dimensional toric code of distance d'''
    Ik = ZMatI(k)
    rd = ZMat([0] * (d-1) + [1])
    ad = ZMat([1] * d)
    temp = []
    for i in range(k):
        kList = [Ik[i]] + [rd] * k
        kList[i+1] = ad
        temp.append(kronList(kList))
    return np.vstack(temp)

def CSSwithLO(target,d):
    '''Return a CSS code with an implementation of a desired logical CP operator using single-qubit phase gates.
    Inputs:
    target: string representing the target CP operator
    d: distance of the toric code the resulting code is based on.
    Output:
    SX, LX and Clifford hierarchy level'''
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
    return SX, LX, t

def codeSearch(target, d, debug=False):
    '''Run the algorithm for generating a CSS code with transversal implementation of a desired logical operator and print output
    Inputs:
    target: string representing the target CP operator
    d: distance of the toric code the resulting code is based on.
    debug: if true, verbose output'''
    ## make a CSS code
    SX, LX, t = CSSwithLO(target,d)
    Eq,SX,LX,SZ,LZ = CSSCode(SX,LX)
    N = 1 << t
    r, n = np.shape(SX)
    compact = n > 15
    if debug:
        print('Embedded Code Checks and Logicals')
        print_SXLX(SX,LX,SZ,LZ,compact)

        ## Logical identities
        K_M = getKM(Eq, SX, LX, N)

        ## Algorithm 1 - search
        print(f'\nKernel Method - Search for {target}')
        ker_search(target,Eq,LX,SX,N,compact)

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


# def CSSwithLO2(target,d):
#     (x,q1), V1, t = Str2CP(target)
#     k = len(x)
#     n = k * d
#     ## identity on k qubits
#     Ik = ZMatI(k)
#     ## repetition code on d qubits
#     Rd = repCode(d)
#     SX = np.kron(Ik,Rd)
#     ## Logical X operators
#     Ld = ZMat2D([0] * (k-1) + [1])
#     LX = np.kron(Ik,Ld)
#     V = Mnt(n,t)
#     SX = matMul(SX,V.T,2)
#     LX = matMul(LX,V.T,2)
#     return SX, LX, t

# def codeSearch2(target, d, debug=False):
#     ## make a CSS code
#     SX, LX, t = CSSwithLO(target,d)
#     Eq,SX,LX,SZ,LZ = CSSCode(SX,LX)
#     N = 1 << t
#     K_M = LIAlgorithm(Eq,LX,SX,N,compact=True,debug=debug)
#     zList = DiagLOComm(SX,K_M,N)
#     ## find level t operators
#     ix = [min(np.gcd(z,N))==1 for z in zList]
#     zList = zList[ix]
#     print(f'level {t} operators')
#     print(ZmatPrint(zList))
#     w = np.sum(zList,axis=0)
#     ix = w > 0
#     SX = SX[:,ix]
#     LX = LX[:,ix]
#     Eq,SX,LX,SZ,LZ = CSSCode(SX,LX)
#     ker_search(target,Eq,LX,SX,t,debug=True)