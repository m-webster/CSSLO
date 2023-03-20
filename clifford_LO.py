from common import *
import itertools as iter
import json
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
    SZ = ZMat(SZ,n)
    LZ = ZMat(LZ,n)
    LX = ZMat(LX,n)
    return Eq,SX,LX,SZ,LZ

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
	
def LXZCanonical(LX, LZ):
    """Modify LX, LZ such that LX LZ^T = I if possible."""
    LX, LZ = LXZDiag(LX,LZ)
    LZ, LX  = LXZDiag(LZ,LX)
    return LX, LZ

def LXZDiag(LX,LZ):
    """Convert LX to a form with minimal overlap with LZ. helper for LXZCanonical"""
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
    A = SZ if len(SX) == 0 else SX
    r,n = np.shape(A)
    return ZMat(SX,n),ZMat(SZ,n),ZMat(SXZ,n)


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
    # triGeom(G)
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



############################################################
## 2D Hyperbolic Tesselations
############################################################


def importCodeList(myfile):
    '''Import hyperbolic surface codes stored in myfile.
    Records in myfile are stored in JSON format.
    Parse each record and return list of dict codeList.'''
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

def hypCellulation(myrow):
    '''For each face of a 2D hyperbolic tesselation, generate list of adjacent vertices and faces in order'''
    ## adjacency matrices
    EV = str2ZMatdelim( myrow['zEV']).T
    FE = str2ZMatdelim( myrow['zEF'])
    FV = str2ZMatdelim( myrow['zFV'])
    ## number of faces, edges, vertices
    E,V = np.shape(EV)
    F,V = np.shape(FV)

    temp = []
    for i in range(F):
        Fv = FV[i]
        Fe = FE[i]
        Fev = EV * Fv 
        Fev = Fev * ZMat([Fe]).T
        ev = {i:set(bin2Set(Fev[i])) for i in range(len(Fev)) if np.sum(Fev[i]) > 0}
        Fev = Fev.T
        ve = {i:set(bin2Set(Fev[i])) for i in range(len(Fev)) if np.sum(Fev[i]) > 0}
        Ffe = FE * Fe
        Ffe[i,:] = 0 
        ef = [bin2Set(v) for v in Ffe.T]
        ef = {i:ef[i][0] for i in range(len(ef)) if len(ef[i]) == 1}       
        vList,fList = [],[]
        v = min(ve.keys())
        while len(ve[v]) > 0:
            e = ve[v].pop()
            ev[e].remove(v)
            v1 = ev[e].pop()
            ve[v1].remove(e)
            vList.append(v)
            fList.append(ef[e])
            v = v1
        temp.append((vList,fList))
    return temp

def PrintCellulation(A):
    '''Print result of the cellulation generated by hypCellulation'''
    temp = []
    for i in range(len(A)):
        temp.append(f'\nFace {i}:')
        temp.append(f'Vertices {A[i][0]}')
        temp.append(f'Faces {A[i][1]}')
    return '\n'.join(temp)

def NColour(A,N):   
    '''Assign colours to faces of 2D hyperbolic tesselation'''
    FAdj = [a[1] for a in A]
    F = np.max([np.max(a) for a in FAdj]) +1
    todo = [(0,-1)]
    visited = set([0])
    colouring = [-1] * F
    while len(todo) > 0:
        curr, prev = todo.pop(0)
        myrow = FAdj[curr]
        if prev < 0:
            ## initial face
            colouring[curr] = 0
            ix = 0
            col = 1
        else:
            ## subsequent faces
            ix = myrow.index(prev)
            col = colouring[prev]
        currCol = colouring[curr]
        V = len(myrow)
        for i in range(V):
            f = myrow[(i + ix) % V]
            if f not in visited:
                visited.add(f)
                colouring[f] = col
                todo.append((f,curr))
            col = (-col - currCol ) % N
    return colouring

def testColouring(A,colouring):
    '''Test colouring generated by NColour'''
    FAdj = [a[1] for a in A]
    for f1 in range(len(FAdj)):
        for f2 in FAdj[f1]:
            if colouring[f1] == colouring[f2]:
                return False 
    return True

def printColouring(colouring):
    '''Print colouring generated by NColour'''
    N = max(colouring) +1
    temp = [[] for i in range(N)]
    for i in range(len(colouring)):
        temp[colouring[i]].append(i)
    for i in range(N):
        print(i,':',temp[i])

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


## build 2D toric code from repetition code and SHPC constr
def toric2D(r):
    '''Generate distance r 2D toric code using SHCP construction.
    Returns SX, SZ.'''
    A = repCode(r,closed=False)
    return SHPC(A)

## partition for toric code SS^3 logical operator
def toric2DPartition(r):
    '''Generate qubit partition for distance r 2D toric code.'''
    A = repCode(r)
    return SHPC_partition(A)

##############################################
## CHAIN COMPLEXES
##############################################

def complexCheck(AList):
    '''Check if AList is a valid complex'''
    for i in range(len(AList)-1):
        Ai = AList[i]
        mi,ni = np.shape(Ai)
        Aj = AList[i+1]
        mj,nj = np.shape(Aj)
        ## check dimension of matrices
        if ni != mj:
            print(f'ni={ni} != mj={mj} for i={i},j={i+1}')
            return False
        ## check that successive operators multiply to zero
        AiAj = matMul(Ai,Aj,2)
        if not np.sum(AiAj) == 0:
            print(f'Ai@Aj != 0 for i={i},j={i+1}')
            return False
    return True

def complexNew(AList):
    '''Make a new complex - make sure there's a zero operator at the end'''
    AList = complexTrim(AList)
    AList = complexAppendZero(AList)
    return AList

def complexTrim(AList):
    '''Remove any all zero matrices from beginning of AList.'''
    temp = []
    i = 0
    while np.sum(AList[i]) == 0:
        i+=1
    return AList[i:]

def complexAppendZero(AList):
    '''Add zero operator to beginning of AList'''
    m,n = np.shape(AList[0])
    return [ZMatZeros((1,m))] + AList

def complexDims(AList):
    '''Return dimensions of each space acted upon by AList.'''
    return [np.shape(A)[1] for A in AList]

def complexTCProd(AList,BList):
    '''Generate check matrices for total complex of product of two chain complexes.'''
    ABList = complexNew(AList), complexNew(BList)
    dimAB = [complexDims(AB) for AB in ABList]
    lenAB = [len(AB) for AB in ABList]
    sList = [[] for i in range(lenAB[0] + lenAB[1] -1)]
    for i in range(lenAB[0]):
        for j in range(lenAB[1]):
            sList[i+j].append((i,j))
    temp = []
    for i in range(1,len(sList)):
        C = []
        for rAB in sList[i-1]:
            myrow = []
            for cAB in sList[i]:
                dAB = ZMat(cAB) - ZMat(rAB)
                kronList=[]
                for k in range(2):
                    r,c,d = rAB[k],cAB[k],dAB[k]
                    cellDim = (dimAB[k][r],dimAB[k][c])
                    if d == 1:
                        AB = ABList[k][c]
                    elif d == 0:
                        AB = ZMatI(cellDim[0])
                    else:
                        AB = ZMatZeros(cellDim)
                    kronList.append(AB)
                myrow.append(np.kron(kronList[0],kronList[1]))
            C.append(np.hstack(myrow))
        C = np.vstack(C)
        temp.append(C)
    return temp

def toricDd(D,d):
    '''Generate SX,SZ for toric code of D dimensions of distance d'''
    H = repCode(d)
    AList = [H.copy()]
    H = complexNew([H])
    for i in range(D-1):
        AList = complexTCProd(H,complexNew(AList))
    SX = AList[-1].T
    r,n = np.shape(SX)
    SZ = AList[-2] if len(AList) > 1 else ZMatZeros((0,n))
    return SX,SZ

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


def nkdReport(SX,LX,SZ,LZ):
    '''Report [[n,k,dX,dZ]] for CSS code specified by SX,LX,SZ,LZ.'''
    k,n = np.shape(LX)
    Zop, Zact = ZDistance(SZ,LZ,LX)
    Xop, Xact = ZDistance(SX,LX,LZ)
    return(f'n:{n} k: {k} dX: {sum(Xop)} dZ:{sum(Zop)}')

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

def print_codewords(Eq,SX,LX):
    '''Print the canonical codwords of a CSS code defined by X-checks SX and X-logicals LX'''
    V, CW = codewords(Eq,SX,LX)
    print('\nCodewords')
    for i in range(len(V)):
        print(f'{ket(V[i])} : {state2str(CW[i])}')

def state2str(S):
    '''Print a state corresponding to a binary matrix S in the form \sum_{x \in S}|x>'''
    return "+".join([ket(x) for x in S])

def ket(x):
    '''Display |x> for states.'''
    return f'|{ZMat2str(x)}>'


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
        c, v = matResidual(K_M, czp, N)
        ## if non-zero, there is an error
        if np.sum(c) > 0:
            print(func_name(),f'Error: x={x},z={z},c={c}')
            return False 
    return True

def CPIsLO(qVec,SX,CPKM,V,N,CP=True):
    '''Check if qVec is a logical operator by calculating the group commutator [[A, CP_V(qVec)]] for each A in SX
    Inputs: 
    qVec: Z_2N vector of length |V| representing a product of CP operators \prod_{v \in V}CP_N(qVec[v],v)
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
        c, u = matResidual(CPKM,c,2*N)
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
            # ix = binInclusion(vList,v)
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
    k,n = np.shape(LX)
    Eq = ZMatZeros((1,n))
    vLX,vList = Orbit2dist(Eq,LX,t,True)
    pVec = ZMat(np.mod([[np.dot(z,x) for x in vLX] for z in LZ],N))
    return pVec,vList

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


def updateSX(SX,SXt,i):
    '''Add row i to all elements of SX - helper function for minWeight'''
    SXw = np.sum(SX,axis=-1)
    e = SX[i]
    SXe = np.mod(SX+e,2)
    SXet = [tuple(e) for e in SXe]
    SXew = np.sum(SXe,axis=-1)
    t = 0
    for j in range(len(SX)):
        if j != i and (SXew[j],SXet[j]) < (SXw[j],SXt[j]):
            SX[j] = SXe[j]
            SXt[j] = SXet[j]
            t+=1
    return SX,SXt,t

def minWeight(SX):
    '''Return a set of vectors spanning <SX> which have minimum weight.'''
    done = False
    w = np.sum(SX)
    SXt = [tuple(e) for e in SX]
    while not done:
        done = True
        for i in range(len(SX)):
            SXi,SXit,t = updateSX(SX,SXt,i)
            # t is the number of times SX was updated by updateSX
            if t > 0:
                SX = SXi
                SXt = SXit
                done = False
    return SX

def ZDistance(SZ,LZ,LX):
    '''Find lowest weight element of <SZ,LZ> which has a non-trivial logical action.
    Uses coset leader method of https://arxiv.org/abs/1211.5568
    Return z component and action'''
    GZ = np.vstack([SZ,LZ])
    GZ, rowops = How(GZ,2)
    GZ =  minWeight(GZ)   
    ## calculate logical actions by multiplying by X-logical matrix
    LA = matMul(GZ,LX.T,2)
    ## nontrivial logical actions
    ix = np.sum(LA,axis=-1) > 0
    ## if len(ix) = 0, distance is min weight of stabilisers
    if np.any(ix):
        GZ = GZ[ix]
        LA = LA[ix]
    ix = np.argmin(np.sum(GZ,axis=-1))
    return GZ[ix], LA[ix]

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

def getKM(Eq,SX, LX,N):
    '''Return KM - Z_N matrix representing phase and z components of diagonal logical identities.'''
    t = log2int(N)
    A = Orbit2dist(Eq, np.vstack([SX,LX]), t)
    A = np.hstack([A,[[1]]*len(A)])
    return getKer(A,N)

########################################################
## Generators of Diagonal Logical XP Group via Kernel Method
########################################################

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

########################################################
## Search for LO by logical action via Kernel Method
########################################################

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
    # print(func_name(), n,r,k, t)
    (x,qL), VL, t2 = Str2CP(target,n=k)
    if t > t2:
        qL = ZMat(qL) * (1 << (t-t2))
    N = 1 << t 
    SXLX = np.vstack([SX,LX])
    EL, uvList = Orbit2dist(Eq,SXLX,t,return_u = True)
    vList = uvList[:,r:]
    pList = np.mod(-ZMat([[CPACT(v,qL,VL,N)] for v in vList] )//2,N)
    KL = getKer(np.hstack([pList, EL]),N)
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

def comm_method(Eq, SX, LX, SZ, t, compact=True, debug=True):
    '''Run the commutator method and print results.
    Inputs:
    Eq: 1xn zero vector
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
        K_M = LIAlgorithm(Eq,LX,SX,N//2,compact,debug=debug) * 2
    ## z components of generators of diagonal XP LO group
    K_L = DiagLOComm(SX,K_M,N)
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
    A, rowops = How(A,N)
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

    ## move highest weight ops to left - more efficient in some cases
    V = np.flip(V,axis=0)
    ## Construct Embedded Code
    SX_V = matMul(SX, V.T, 2)
    LX_V = matMul(LX, V.T, 2)
    Eq_V = ZMatZeros((1,len(V))) 
    SZ_V = None
    ## Find diagonal logical operators at level t
    K_L, qList, VL, K_M = comm_method(Eq_V, SX_V, LX_V, SZ_V, t, compact=False, debug=False)
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
    CPKL, rowops = How(CPKL + CPKM,2*N)
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
        CPKM, rowops = How(CPKM, 2*N)
        print('LO Test:',CPIsLO(qCP,SX,CPKM,V,N))
        if len(LX) < 5:
            print('Testing Action on Codewords')
            codeword_test(qCP,Eq,SX,LX,V,N)
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
        qVec1 = HowRes(KL[:,ix], qVec[ix],2*N)
        ## s is the support of z1
        s = bin2Set(qVec1)
        m = min(s)
        ## if m < LR[0] then it was not eliminated - invalid configuration
        if m >= len(LRix[0]): 
            ## indices corresp to original col order
            ixR = ixRev(ix)
            # print('overlapCount',overlapCount(qVec1, V))
            ## reorder z1
            qVec1 = qVec1[ixR] 
            # oc, oix = overlapCount(qVec1, V)
            oc = overlapCount(qVec1, V)
            # ## original order of m
            oix = ix[m] 
            ## test for depth-one operator:
            if oc == 0:
                print(func_name(),'Visited',len(visited),'todo',len(todo))
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
    SX, rowops = How(SX[:,ix],2)
    LX, rowops = How(LX[:,ix],2)
    ## sort cols of SX/LX by weight of SX then alpha SX then weight LX then alpha LX
    cList = [(sum(SX[:,i]),tuple(SX[:,i]),sum(LX[:,i]),tuple(LX[:,i])) for i in range(n)]
    ix = argsort(cList)
    ## Flip row order
    LX = np.flip(LX[:,ix],axis=0)
    SX = np.flip(SX[:,ix],axis=0)
    return SX,LX


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
    Eq,SX,LX,SZ,LZ = CSSCode(SX,SZ=SZ)
    ## Get Canonical LO
    (qCP,VCP), (qRP,V) = canonical_logical_algorithm(qL,VL,LZ,t,debug=False)

    ## sort V so that cols with lowest gcd are to the bottom
    g = np.gcd(qRP//2,N)
    maxg = max(np.mod(g,N))
    if maxg > 1:
        ix = argsort(g,reverse=True)
        V = V[ix]
        qRP = qRP[ix]
    ## make embedded code
    SXV = matMul(SX,V.T,2)
    LXV = matMul(LX,V.T,2)
    EqV = ZMatZeros((1,len(V)))
    ## z component of transversal LO on embedded code
    z = qRP//2
    ## simplify z component
    if maxg > 1 and t > 1:
        K_M = getKM(EqV,SXV,LXV,N//2)[:,:-1] * 2
        z, u = matResidual(K_M, z,N)
    ## remove qubits where z[i] = 0, and make SX/LX into a nice format
    SX,LX = CSSPuncture(SXV,LXV,z)
    return SX,LX


def codeSearch(target, d, debug=False):
    '''Run the algorithm for generating a CSS code with transversal implementation of a desired logical operator and print output
    Inputs:
    target: string representing the target CP operator
    d: distance of the toric code the resulting code is based on.
    debug: if true, verbose output'''
    ## make a CSS code
    SX,LX = CSSwithLO(target,d)
    Eq,SX,LX,SZ,LZ = CSSCode(SX,LX)
    ## calculate distance
    z,pVec = ZDistance(SZ,LZ,LX)
    dZ = np.sum(z)
    x,pVec = ZDistance(SX,LX,LZ)
    dX = np.sum(x)
    return SX,LX, dX,dZ