import itertools as iter
import json
from common import *
from NHow import *
from XCP_algebra import *

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

def HypercubeCells(r):
    """Create Hypercube cells in r dimension on 2^r qubits.
    Returns SX.
    """
    n = 1 << r
    SX = []
    a, b = 1, n >> 1
    for i in range(r):
        for j in range(2):
            x = ([j] * a + [1-j]*a) * b
            SX.append(x)
        a = a << 1
        b = b >> 1
    return ZMat(SX)

def punctureRem(SX,ix):
    '''Remove qubits indexed by ix plus any stabilisers with support on these'''
    if ix is None:
        return SX
    r,n = np.shape(SX)
    ixr = sorted(set(range(n)) - set(ix))
    w = np.sum(SX[:,ix],axis=-1) == 0
    SX = SX[w]
    return SX[:,ixr]

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
    SX = getH(G,2)
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
## Symmetric Hypergraph Product Codes
############################################################

## Symmetric Hypergraph Product Code
def SHPC(T):
    '''Make symmetric hypergraph product code from T.
    T can either be a string or np array.
    Returns SX, SZ.'''
    T = bin2ZMat(T)
    H = matMul(T.T, T,2)
    return HPC(H,H)

def HPC(A,B):
    '''Make hypergraph product code from clasical codes A, B
    A and B can either be a string or np array.
    Returns SX, SZ.'''
    A = bin2ZMat(A)
    B = bin2ZMat(B)
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
    T = bin2ZMat(T)
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

def circulantMat(v):
    '''Circulant matrix formed from row vector v'''
    temp = [np.roll(v,i) for i in range(len(v))]
    return ZMat(temp)

def figure8(L):
    '''Figure 8 graph - 2 cycles of length L meeting in a common point'''
    H = repCode(L)
    ZL = ZMatZeros((L,L-1))
    return np.vstack([np.hstack([H,ZL]),np.hstack([ZL,H])]).T


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


###################################
## Reflection Group Complexes
###################################

def RG2Complex(myrow):
    '''convert string format to RG complex boundary operators'''
    boundaryOperators = []
    ## read boundary maps for each level of the complex
    for myLabel in [f'Z{i}' for i in range(4,-1,-1)]:
        if myLabel in myrow: 
            ## convert to boundary operators
            boundaryOperators.append(str2ZMatdelim(myrow[myLabel]))
    return complexNew(boundaryOperators)

def importRGList(myfile):
    '''Import hyperbolic surface codes stored in myfile.
    Records in myfile are stored in JSON format.
    Parse each record and return list of dict codeList.'''
    mypath = sys.path[0] + "/hyperbolic_codes/"
    f = open(mypath + myfile, "r")
    mytext = f.read()
    mytext = mytext.replace("\\\n","").replace("\n","")
    mytext = mytext.replace("{","\n{")
    mytext = mytext.split("\n")
    codeList = []
    for myline in mytext:
        if len(myline) > 0 and myline[0] != "#":
            myrow = json.loads(myline)
            codeList.append([myrow['index'],RG2Complex(myrow)])
    f.close()  
    return codeList

def printRGList(codeList,myfile,checkValid=False):
    '''Print parameters of the hyperbolic codes stored in list of dict codeList'''
    temp = []
    temp.append(f'Codes in File {myfile}:\n')
    valTxt = "\tValid" if checkValid else ""
    D = len(codeList[0][1]) 
    myrow = f'i\tindex{valTxt}'
    for i in range(D):
        myrow += f'\t|C{i}|'
    temp.append(myrow)
    for i in range(len(codeList)):
        myrow = codeList[i]
        ix = myrow[0]
        C = myrow[1]
        rowDesc = [i,ix]
        if checkValid:
            rowDesc += [complexCheck(C)]
        rowDesc += complexDims(C)
        temp.append("\t".join([str(a) for a in rowDesc]))
    return "\n".join(temp)

def complexCProduct(C):
    if len(C) == 0:
        return []
    P = C[0]
    temp = [P.T]
    for i in range(1,len(C)):
        P = mod1(P @ C[i])
        temp.append(P.T)
    return temp

def complex2ColourCode(C):
    '''Make a colour code from Complex
    Qubits: vertices
    SZ: 2-faces
    SX: D-faces'''
    ## express complex in terms of adjacecy matrices wrt 0-cells (vertices)
    C1 = complexCProduct(C[1:])
    ## 2D Faces
    SZ = C1[1]
    ## highest dim cells
    SX = C1[-1]
    return SX, SZ

def complex2SurfaceCode(C):
    '''Make a surface code from Complex
    Qubits: edges
    SZ: plaquettes
    SX: vertices'''
    ## express complex in terms of adjacecy matrices wrt 1-cells (edges)
    ## vertex operators
    SX = C[1]
    ## plaquette operators
    SZ = C[2].T
    return SX, SZ

def hypCellulation(C):
    '''For each face of a 2D hyperbolic tesselation, generate list of adjacent vertices and faces in order'''
    ## adjacency matrices
    VE = C[1]
    FE = C[2]
    FV = mod1(C[1] @ C[2])
    ## number of faces, edges, vertices
    E,V = np.shape(VE)
    F,V = np.shape(FV)

    temp = []
    for i in range(F):
        Fv = FV[i]
        Fe = FE[i]
        Fev = VE * Fv 
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