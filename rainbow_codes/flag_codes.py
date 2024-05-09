from add_parent_dir import *
from common import *
from NHow import *
from CSSLO import *
import itertools as iter
from queue import PriorityQueue
import concurrent.futures


#####################################
## Standard Flag Code Types
#####################################


def Complex_to_Code_MSG(C,SG=False,CP=False):
    '''SX and SZ are MSG'''
    IdC = None
    d = len(C)
    xCycles = iter.combinations(range(d+1),d)
    Sxmc = [('x','m',c) for c in xCycles]
    zCycles = iter.combinations(range(d+1),2)
    Sxmc += [('z','m',c) for c in zCycles]
    return flag_code(C,Sxmc,SG=SG,IdC=IdC,CP=CP)

def Complex_to_Code_Mixed(C,SG=False,CP=True):
    '''SX and SZ are mixed MSG and RSG types'''
    IdC = None
    d = len(C)
    RIncl = {0,d}
    xCycles = iter.combinations(range(d+1),d)
    Sxmc = [('x','m' if RIncl.issubset(c) else 'r',c) for c in xCycles]
    zCycles = iter.combinations(range(d+1),2)
    Sxmc += [('z','m' if RIncl.issubset(c) else 'r',c) for c in zCycles]
    return flag_code(C,Sxmc,SG=SG,IdC=IdC,CP=CP)

def Complex_to_Code_RSG(C,SG=False,CP=False):
    '''SX: MSG; SZ: RSG'''
    IdC = None
    d = len(C)
    xCycles = iter.combinations(range(d+1),d)
    Sxmc = [('x','m' if d > 2 else 'r',c) for c in xCycles]
    zCycles = iter.combinations(range(d+1),2)
    Sxmc += [('z','r',c) for c in zCycles]
    return flag_code(C,Sxmc,SG=SG,IdC=IdC,CP=CP)

def Complex_to_PinCode(C,SG=False,CP=False):
    '''SX: MSG; SZ: MSG. Add boundary pins to ensure commutation relations'''
    d = len(C)
    IdC = None
    ## Add extra pin to C[0] if required
    C[0] = make_even(C[0])
    ## Add extra pin to C[D] if required
    C[-1] = make_even(C[-1].T).T
    xCycles = iter.combinations(range(d+1),d)
    Sxmc = [('x','m',c) for c in xCycles]
    zCycles = iter.combinations(range(d+1),2)
    Sxmc += [('z','m',c) for c in zCycles]
    return flag_code(C,Sxmc,SG=SG,IdC=IdC,CP=CP)

def Complex_to_24_Cell(C,SG=False,CP=False):
    '''SX and SZ are MSG; edge contraction to reduce qubit numbers - 24 cells for repetition codes'''
    IdC = [0,3]
    xCycles = [(0,1,2),(1,2,3)]
    Sxmc = [('x','m',c) for c in xCycles]
    zCycles = [(0,1),(1,2),(2,3)]
    Sxmc += [('z','m',c) for c in zCycles]
    return flag_code(C,Sxmc,SG=SG,IdC=IdC,CP=CP)

def Complex_to_8_24_48_Cell(C,SG=False,CP=False):
    '''SX and SZ are MSG; edge contraction to reduce qubit numbers'''
    IdC = [3]
    xCycles = [(0,1,2),(0,2,3),(1,2,3)]
    Sxmc = [('x','m',c) for c in xCycles]
    zCycles = [(0,1),(0,2),(1,2),(0,1,3),(2,3)]
    Sxmc += [('z','m',c) for c in zCycles]
    return flag_code(C,Sxmc,SG=SG,IdC=IdC,CP=CP)


#####################################
## Generate Flag Codes
#####################################

def flag_code(C,Sxmc,SG=False,IdC=None,CP=True):
    '''Return Flag code in binary format
    C: a list of boundary operators representing a complex
    Sxmc: list of stabiliser types encoded as tuples (x,m,c): 
    - x is either 'x' or 'z' indicating whether this is an X or Z-stabiliser
    - m is either 'r' or 'm' indicating rainbow or maximal subgraph
    - c is a list of colours for the subgraphs
    SG: if True, return a min weight set of generators of each cycle type - this can be slow
    IdC: identify qubits within specified maxmimal subgraphs- eg (0,1) identifies qubits in the same maximal (0,1) subgraph'''
    ## Check if this is a valid complex
    checkOK = complexCheck(C)
    if not checkOK:
        return [ZMatZeros((0,0)) for i in range(4)]
    ## generate flags from complex C
    F = complex2Flag(C)
    n = len(F)
    ## construct Flag Graph - results in a list of adjacency dictionaries - one for each colour
    FG = flagGraph(F)
    subGraphs = dict()
    SX_deferred = set()
    SG_now = set()
    n = len(FG[0])
    ## split into list of subgraphs to be calculated immediately (SG_now) or later (SX_deferred)
    for x,m,colours in sorted(Sxmc,reverse=True):
        cols = set2tuple(colours)
        ix = (len(cols),m,cols)
        ## calculate now if a Z-check or MSG-type
        if (x == 'z') or (m == 'm'):
            SG_now.add(ix)
            if m == 'r':
                SG_now.add((len(cols),'m',cols))
        else:
            SX_deferred.add(ix)
    ## Calculate SG of types in calc_now
    for l,m,cols in sorted(SG_now):
        CC = getSG(FG,cols, m, subGraphs)
    ## Construct lists of cycles corresponding to stabilisers of X and Z-type
    ## initialise X and Z stabiliser lists
    SX = []
    SZ = []
    ## Check if we have already calculated subgraphs of the required type - if so, add to SX or SZ as appropriate
    for x,m,colours in sorted(Sxmc,reverse=True):
        cols = set2tuple(colours)
        ix = (cols,m)
        if ix in subGraphs:
            CC = subGraphs[ix]
            Stype = SX if x == 'x' else SZ
            Stype.extend(CC)
    ## Identify qubits via MSG specified by IdC
    IdM = getSG(FG,IdC,'m',subGraphs) if IdC is not None else None
    ## Convert SX,SZ to binary matrices and identify qubits as appropriate
    SZ = identifyQubits(IdM,Sets2ZMat(n,SZ))
    SX = identifyQubits(IdM,Sets2ZMat(n,SX))
    ## Calculate the balance of the X-checks as well as LX, LZ
    SX,LX,LZ = flagSXLXLZ(SZ,SX,FG,subGraphs,SX_deferred,IdM)
    ## Optionally simplify generators
    if SG:
        SX,p = lowWeightGens(SX,N=2)
        SZ,p = lowWeightGens(SZ,N=2)
    ## Optionally find Coloured Paulis (slow for large codes)
    if CP:
        ## split into Z and X-type stabilisers
        SX_types = []
        SZ_types = []
        for x,m,col in Sxmc:
            SType = SX_types if x == 'x' else SZ_types
            ix = set2tuple(col)
            SType.append(ix)
        ## Update LX and LZ
        LX = coloured_LX(FG,SZ_types,subGraphs,SX,LX,IdM)
        LZ = coloured_LX(FG,SX_types,subGraphs,SZ,LZ,IdM)
    return SX,SZ,LX,LZ

def coloured_LX(FG,SZ_types,subGraphs,SX,LX,IdM):
    '''Calculate Coloured logicals'''
    SXLX = np.vstack([SX,LX])
    ## intialise LX1
    LX1 = []
    ## iterate through each type of SZ and get LX
    for colz in SZ_types:
        LX1.append(get_col_LX(colz,FG,subGraphs,SX,SXLX,IdM))
    LX1 = np.vstack(LX1)
    ## get low weight independent set
    LX1 = lowWeightGens(LX1,S=SX,N=2)
    return LX1

def get_col_LX(colz,FG,subGraphs,SX,SXLX,IdM):
    D = len(FG)
    n = len(FG[0])
    ## invert colour set
    colz_inv = set2tuple(set(range(D)).difference(colz))
    ## MSG of type colz_inv
    C1 = getSG(FG,colz_inv,'m',subGraphs)
    CX = identifyQubits(IdM,Sets2ZMat(n,C1))
    ## intersect linear combinations of MSG with SXLX
    LX1 = nsIntersection([CX,SXLX],2)
    L,H = HowRes(SX,LX1,2)
    ## stabilisers are all zero vectors after taking HowRes - exclude these
    w = np.sum(L,axis=-1)
    return LX1[w > 0,:]

def flagSXLXLZ(SZ,SX,FG,subGraphs,SXtodo,IdM):
    '''Calculate additional SX via faster Kernel method, LX and LZ'''
    ## RREF of SZ - ixs are the pivots of the resulting matrix
    HZ,ixs = getH(SZ,2,retPivots=True)
    s,n = HZ.shape
    ## reorder cols of SX so that ixs are on RHS
    ix = np.hstack([invRange(n,ixs), inRange(n,ixs)])
    ## RREF of SX - ixr are the pivots of the resulting matrix
    HX,ixr = getH(ZMat(SX)[:,ix],2,retPivots=True)
    ## restore HX and ixr to original order
    ixR = ixRev(ix)
    HX = HX[:,ixR]
    ixr = [ix[i] for i in ixr]
    ## there are k remaining columns not in ixr or ixs - ixk
    ixk = invRange(n,ZMat(set(ixr).union(ixs)))
    k = len(ixk)
    ## Calculate partial LX: of form (E.T|0|I) with cols in order (ixs|ixr|ixk)
    LX = ZMatZeros((k,n))
    E = HZ[:,ixk]
    LX[:,ixs] = E.T
    for i in range(k):
        LX[i,ixk[i]] = 1
    ## Calculate additional subgraphs for X-checks via kernel method
    if len(SXtodo) > 0:
        LXSX = np.vstack([SX,LX])
        tostack = [SX]
        for l,m,cols in sorted(SXtodo):
            CC = RSGKer(FG,cols,subGraphs,LXSX,IdM)
            ## previous method - much slower for 3+ colour RSG
            # CC = getRSG(FG,cols,subGraphs)
            # CC = Sets2ZMat(n,list(CC))
            tostack.append(CC)
        ## Append additional SX types and calculate RREF - ixr are the pivots of HX
        SX = np.vstack(tostack)
        HX,ixr = getH(ZMat(SX)[:,ix],2,retPivots=True)
        ## restore HX and ixr to original order
        HX = HX[:,ixR]
        ixr = [ix[i] for i in ixr]
        ## ixk are the remaining columns excluding ixr and ixs
        ixk = invRange(n,ZMat(set(ixr).union(ixs)))
        k = len(ixk)
        ## Update LX for new SX
        LX = ZMatZeros((k,n))
        E = HZ[:,ixk]
        LX[:,ixs] = E.T
        for i in range(k):
            LX[i,ixk[i]] = 1
    ## Calculate LX: of form (0|A.T|I) with cols in order (ixs|ixr|ixk)
    LZ = ZMatZeros((k,n))
    A = HX[:,ixk]
    LZ[:,ixr] = A.T 
    for i in range(k):
        LZ[i,ixk[i]] = 1    
    ## return updated SX and LX/LZ
    return SX,LX,LZ

def identifyQubits(M,A):
    '''Identify qubits in MSG list specified by M'''
    if M is None or len(M) == 0:
        return A
    temp = ZMatZeros((len(A),len(M)))
    for i in range(len(M)):
        ix = sorted(M[i])
        mycol = np.sum(A[:,ix],axis=-1)
        mycol[mycol > 1] = 1
        temp[:,i] = mycol
    return temp


#####################################
## Product complexes
#####################################

def productComplex(HList):
    '''Take a list of binary matrices and return a product complex'''
    d = len(HList)
    C = complexNew([HList[0]])
    for i in range(1,d):
        Ci = complexNew([HList[i]])
        C = complexTCProd(C, Ci)
    ## reverse order and transpose to match Arthur's code
    C = [c.T for c in reversed(C)]
    return C

def doubleComplex(C):
    C[0] = np.vstack([C[0],C[0]])
    return C

#################################
## Generate Flags
#################################

def flagRecurse(C,d,i,s):
    '''form flags of length d+1 from flag of length d
    C is a set of boundary operators of a binary complex
    C[d][i] is boundary of dimension d
    s is a flag of length d - 1, not including i'''
    ## append i to the flag s
    s = s + [i]
    
    if d  == len(C):
        ## if we are at the final boundary, return the flag
        return [s]
    else:
        ## otherwise, extend the flag by looking at the next boundary operator
        temp = []
        for j in bin2Set(C[d][i]):
            temp.extend(flagRecurse(C,d+1,j,s))
        return temp

def complex2Flag(C):
    '''generate flags for the binary complex C'''
    temp = []
    ## iterate through all rows in the first boundary operator
    for i in range(len(C[0])):
        temp.extend(flagRecurse(C,0,i,[]))
    ## make set of tuples to elimnate duplicates
    temp = {tuple(s) for s in temp}
    return ZMat(sorted(temp))

def flagAdj(F,i):
    '''make adjacency matrix from set of flags F
    flags are adjacent if they are the same, apart from component i'''
    n = len(F)
    A = [set() for i in range(n)]
    adjDict = dict()
    for j in range(n):
        ## k is the flag with component i cleared
        k = tuple(np.hstack([F[j,:i],F[j,i+1:]])) 
        if k not in adjDict:
            adjDict[k] = []
        adjDict[k].append(j)
    for k,FList in adjDict.items():
        ## take pairs of flags which have the same k, and make adjacency matrix
        for (a,b) in iter.combinations(FList,2):
            A[a].add(b)
            A[b].add(a)
    return A

def flagGraph(F):
    '''make adjacency matrix from set of flags F
    There is an adjacency matrix for each dimension of the complex'''
    d = len(F[0])
    return [flagAdj(F,i) for i in range(d)]


#################################
## Graph algrorithms
#################################

def getSG(FG,colours,SGType,SGList):
    '''wrapper function for generating maximal and rainbow subgraphs
    FG: flag graph
    colours: colours for the subgraph
    SGType: either m or r'''
    ## check if we have already computed the Subgraph type
    ix = (set2tuple(colours),SGType)
    if ix in SGList:
        return SGList[ix]
    ## If not, calculate the subgraph
    temp = getRSG(FG,colours,SGList) if SGType=='r' else getMSG(FG,colours)
    ## update SGList
    SGList[ix] = temp
    return temp

#################################
## Maximal Subgraphs
#################################

def getMSG(FG,colours):
    '''Generate set of maximal subgraphs for the flag graph FG and set of colours/dimensions colours'''
    ## todo is a list of flags to be explored
    todo = set(range(len(FG[0])))
    temp = []
    while len(todo) > 0:
        i = todo.pop()
        if len(FG[colours[0]][i]) > 0:
            res = MSGExplore(FG,colours,i)
            temp.append(set2tuple(res))
            todo = todo - res
    return temp 

def MSGExplore(FG,colours,i):
    '''explore FG using adjacency types in colours from starting point i'''
    todo = [i]
    visited = {i}
    while len(todo) > 0:
        i = todo.pop()
        for d in colours:
            for j in FG[d][i]:
                if j not in visited:
                    todo.append(j)
                    visited.add(j)
    return visited


#################################
## Rainbow Subgraphs
#################################

def getRSG(FG,colours,SGList):
    '''Generate RSG'''
    if len(colours) == 1:
    # 1-RSG
        M = getSG(FG,colours,'m',SGList)
        return set().union(*[iter.combinations(m,2) for m in M])
    temp = set()
    if len(colours) == 2:      
    # 2-RSG
        return getRSG2(FG,colours,SGList)
    if len(colours) >= 3:
    # 3+ colour-RSG
        return getRSG3Plus(FG,colours,SGList)
    return temp 

def RSGKer(FG,colours,SGList,L,IdM):
    '''Kernel method - faster method for RSG with 3+ colours'''
    # Get MSG corresponding to colours
    MList = getSG(FG,colours,'m',SGList)
    n = len(FG[0])
    MList = identifyQubits(IdM,Sets2ZMat(n,MList))
    n = MList.shape[-1]
    temp = []
    ## for each MSG M
    for M in MList:
        M = bin2Set(M)
        ## move flags in M to the RHS
        ix = np.hstack([invRange(n,M), inRange(n,M)])
        ixR = ixRev(ix)
        A = L[:,ix]
        ## RREF
        A,LI = getH(A,2,retPivots=True)
        ## elements in the span of A which are in the support of M
        ix = [i for i in range(len(LI)) if LI[i] >= n - len(M)]
        temp.append(A[:,ixR].take(indices=ix,axis=0))
    return np.vstack(temp)

def getRSG2(FG,colours,SGList):
    '''get two colour RSGs'''
    # get 1-MSG for 2nd colour 
    F = getSG(FG,colours[1:],'m',SGList)
    # fix order of F
    F = list(F)
    d = colours[0]
    A = FG[d]
    ## adj matrix between 1-faces and flags
    AVF = [{i for i in range(len(F)) if j in F[i]} for j in range(len(A))]
    ## faces still to visit
    faces2visit = set(range(len(F)))
    ## set of 2-RSGs
    temp = set()
    while len(faces2visit) > 0:
        ## next flag to visit
        i = faces2visit.pop()
        ## spanning tree from i - yield tree and unvisited edges
        parentTree, cycles = ST(F,FG,d,colours,AVF,i)
        ## get induced cycles and update temp
        CC = inducedCycles(parentTree, cycles)
        temp.update(CC)
        ## update faces still to visit
        faces2visit.difference_update(parentTree.keys())
    return temp

def nextFaces(F,A,AVF,f1,visitedFlags):
    ## Get next faces for Spanning Tree function ST
    for v1 in F[f1] :
        if v1 not in visitedFlags:
            for v2 in A[v1]:
                if v2 not in visitedFlags:
                    for f2 in AVF[v2]:
                        yield (f2,(v1,v2))

def ST(F,FG,d,colours,AVF,f1):
    '''Generate spanning tree and unvisited edges corresponding to cycles
    FG: flag graph
    ix: ordered colours
    i: index of starting flag'''
    visitedFlags = set()
    cycles = dict()
    todo = [f1]
    parentTree = {f1:None}
    while len(todo) > 0:
        ## i - flag index; s - sequence of colours to visit
        f1 = todo.pop()
        myPath = getPath(parentTree,f1,edgewise=False)
        myPath = [j for i in myPath for j in F[i] ]
        for (f2,e) in nextFaces(F,FG[d],AVF,f1,visitedFlags):
            res = checkRSGSol(myPath + list(F[f2]),colours,FG) if len(colours) > 2 else True
            if res is not False:
                visitedFlags.add(e[0])
                visitedFlags.add(e[1])
                if (f2 not in parentTree):
                    ## j not previously visited - update parentTree and todo
                    parentTree[f2] = (f1,e)
                    todo.append(f2)
                else:
                    cix = set2tuple((f1,f2))
                    if (cix not in cycles):
                        cycles[cix] = set()
                    ## otherwise, add to cycles
                    cycles[cix].add(set2tuple(e))
    return parentTree,cycles

def path2cycle(p1,p2):
    '''form a cycle from path up to i, joined to p[-1] via the path p'''   
    ## find first place where p1 and p2 vary, going from L to R (avoid subcycles)
    n = min(len(p1),len(p2))
    i = 0
    while i < n and p1[i] == p2[i]:
        i+=1
    ## join the paths to form a cycle
    temp = list(p1[i:])  + list(reversed(p2[i:]))
    return temp

def getPath(parentTree,i,edgewise=True):
    '''Return path of flags visited to reach flag i'''
    temp = []
    while parentTree[i] is not None:
        j,e = parentTree[i]
        if edgewise:
            temp.extend([e[1],e[0]])
        else:
            temp.append(j)
        i = j
    temp.reverse()
    return temp

def inducedCycles(parentTree, cycles):
    '''generating set of cycles for connected component corresponding to parentTree'''
    temp = []
    for (i,j),eList in cycles.items():
        p1 = getPath(parentTree,i)
        p2 = getPath(parentTree,j)
        c = path2cycle(p1,p2)
        for e in eList:
            temp.append(list(c) + list(e))
    return {tuple(c) for c in temp}

############################################################
## Checking for valid RSG
############################################################

def checkRSGSol(c,colours,FG):
    '''check if c is a valid RSG. If so, return True. If there are colour duplicates, return False. Otherwise, return measure of how close we are to an RSG'''
    res = checkRSG(c,colours,FG,True)
    # if res is False, then there's a vertex linked twice by a colur
    if res is False:
        return False
    # if there are no missing colours, we have a solution
    if len(res) == 0:
        return True 
    ## order by proportion of vertices which are full colour, the size of c and finally res
    return len(res)/len(c),len(c),res

def checkRSGNode(i,c,colours,FG):
    '''For node i in the subgraph c, check how many neighbours there are of each colour in colours.
    If there are more than one neighbour of a given colour, return False
    Otherwise, return the number of colours for which there is no neighbour'''
    missing_colours = set()
    for d in colours:
        ## check if how many neighbours i has in c of type d
        s = len(FG[d][i].intersection(c))
        ## more than one neighbour of this colour - return False
        if s > 1:
            return False
        ## otherwise, increment number of colours satisfied
        if s == 0:
            missing_colours.add((i,d))
    ## return number of colours still to be satisfied
    return missing_colours

def checkRSG(c,colours,FG,giveMissing=False):
    '''For nodes in the subgraph c, check how many neighbours there are of each colour in colours.
    If there are more than one neighbour of a given colour, return False
    Otherwise, return the number of colours for which there is no neighbour'''    
    missing_colours = set()
    for i in c:
        res = checkRSGNode(i,c,colours,FG)
        if res is False:
            return False
        missing_colours.update(res)
    return missing_colours if giveMissing else len(missing_colours)

##############################################################
## 3+ Colour RSG
##############################################################

def getRSG3Plus(FG,colours,SGList):
    '''Generate RSG for 3+ colours'''
    ## list of RSG to return
    temp = set()
    ## MSG are connected components
    MList = getSG(FG,colours,'m',SGList)
    ## iterate through connected components
    for M in MList:
        tt = set()
        # check if the MSG is an RSG
        res = checkRSGSol(M,colours,FG) 
        if res is True:
            tt.add(M)
        else:
            ## find optimal type of RSG to consider for each M
            M = set(M)
            optList = []
            ## iterate through all choices colour sets with one less element
            for i in range(len(colours)):
                d = colours[i]
                coli = set2tuple([c for c in colours if c != d ])
                ## get RSG with one less colour
                CC = getSG(FG,coli,'r',SGList)
                CC = [c for c in CC if M.issuperset(c)]
                ## make the cycle span
                CC = RSG_span(CC,coli,FG)
                nCC = len(CC)
                optList.append((nCC,d,coli,CC))
            ## work with the minimum number of CC
            nCC,d,coli,CC = min(optList)
            FRSG = Faces2RSG(CC,FG,colours)
            tt.update(FRSG)
        temp.update(tt)   
    return temp 



# def faceCycles(F,FG,d,colours):
#     '''Find cycles made from Faces F with adjacency given by A'''
#     F = list(F)
#     temp = []
#     for parentTree, cycles in spanningTrees(F,FG,d,colours):
#         if len(colours) == 2:
#             CC = inducedCycles(parentTree, cycles)
#         else:
#             CC = FC(parentTree,cycles,F,colours,FG)
#         temp.extend(CC)
#     return temp


def Faces2RSG_Ex(CC,FG,colours):
    '''Exhaustive method for merging faces into RSG'''
    tt =set()
    w = min([len(f) for f in CC])
    maxFlen = 49//w
    CC = list(CC)            
    # try all possible combinations of up to maxFlen faces
    m = min(maxFlen,len(CC))
    for w in range(2,m+1):
        for ix in iter.combinations(range(len(CC)),w):
            ## make a cycle
            c = set().union(*[set(CC[i]) for i in ix])
            c = set2tuple(c)
            ## check if it's a valid RSG
            res = checkRSGSol(c,colours,FG)
            if res is True:
                tt.add(c)
    return tt

def Faces2RSG(F,FG,colours):
    '''Merge Faces F into an RSG. FG is a flag graph'''
    ## Check if whole set of vertices is a solution
    c = set2tuple(set().union(*[f for f in F]))
    res = checkRSGSol(c,colours,FG)
    if res is True:
        return [c]
    ## A is an adjacency matrix for the colours we are searching for
    A = []
    for i in range(len(FG[0])):
        S = set()
        S.update(*[FG[j][i] for j in colours])
        A.append(S)
    ## each element of F can be in at most Fmax combinations
    Fcount = [0 for f in F]
    Fmax = 8
    # max number of vertices in a face
    maxV = 50
    ## list of completed rainbow subgraphs
    temp = set()
    ## convert F to list of tuples
    F = list({set2tuple(c) for c in F})
    ## calculate how many flags have all colours
    missing_colours = [checkRSG(c,colours, FG) for c in F]
    ## order in increasing ratio of full colourings
    ix = argsort([missing_colours[i]/len(F[i]) for i in range(len(missing_colours))],reverse=True)
    ## startingPoints are cycles which are valid but not closed
    startingPoints = []
    for i in ix:
        ## check if cycle is valid
        if missing_colours[i] is not False:
            ## check if cycle is closed
            if missing_colours[i] == 0:
                ## if closed, add to result
                temp.add(F[i])
            else:
                ## otherwise, add to startingpoints
                startingPoints.append(F[i])
    ## maintain list of F-configuations already examined
    visited = set()
    ## iterate through startingpoints
    for i in range(len(startingPoints)):
        solFound = False
        # check if we've already used F max number of times
        if Fcount[i] < Fmax:
            ## ix is a list of faces we include
            ix = (i,)
            ## check if ix is a solution
            c = set2tuple(set().union(*[startingPoints[i] for i in ix]))
            res = checkRSGSol(c,colours,FG)
            if res is True:
                solFound = True
            elif res is not False:
                ## initialise search
                visited.add(ix)
                todo = PriorityQueue()
                todo.put((res,ix))
                ## repeat until either todo is empty or we find a solution
                while todo.qsize() > 0 and not solFound:
                    (s,l,res), ix = todo.get()
                    # vertices we could link to from ix to meet colour deficiency
                    missingAdj = set.union(*[FG[d][i] for (i,d) in res])
                    # convert this to indices of startingPoint faces
                    missingAdj = {i for i in range(len(startingPoints)) if len(missingAdj.intersection(startingPoints[i])) > 0}
                    # make sure we don't add a face which is already in ix
                    for j in missingAdj - set(ix):
                        # also make sure we don't use the face more than max allowed amount
                        if Fcount[j] < Fmax and not solFound:
                            # append the face index
                            ix2 = set2tuple(list(ix) + [j])
                            c = set2tuple(set().union(*[startingPoints[i] for i in ix2]))
                            # check if c already found as a solution
                            if c not in temp and ix2 not in visited and len(c) < maxV:
                                # check for vertices which have missing colours
                                res = checkRSGSol(c,colours,FG)
                                if res is True:
                                    solFound = True
                                # if res is False, then there's a vertex linked twice by a colur
                                elif res is not False:
                                    # if there are no missing colours, we have a solution
                                    todo.put((res,ix2))
                                    visited.add(ix2)
        if solFound:
            # add to list of solutions
            temp.add(c)
            # increment Fcount
            for i in range(len(startingPoints)):
                if set(startingPoints[i]).issubset(c):
                    Fcount[i] += 1
    return temp

def RSG_span(CC,colours,FG):
    '''generate all RSG from cycle basis CC'''
    # make adjacency matric from colour transitions
    A = []
    for i in range(len(FG[0])):
        S = set()
        S.update(*[FG[j][i] for j in colours])
        A.append(S)
    # temp - all cycles (result of function)
    temp = {set2tuple(c) for c in CC}
    # todo - intialised as all elements of cycle basis
    todo = {set2tuple(c) for c in CC}
    visited = set()
    while len(todo) > 0:
        # next item - convert to set
        a = todo.pop()
        a = set(a)
        toAdd = set()
        for b in temp:
            # check if the cycles overlap
            if len(a.intersection(b)) > 0:
                # add the cycles and make a tuple
                c = a.symmetric_difference(b)
                ctup = set2tuple(c)
                # check if we have previously visited this option
                if len(c) > 0 and ctup not in visited:
                    visited.add(ctup)
                    # check if it's a valid RSG
                    res = checkRSG(c,colours,FG)
                    if res is not False and res == 0:
                        # limit FG to vertices in c
                        FGMod = [[c.intersection(FF[i]) if i in c else set() for i in range(len(FF))] for FF in FG]
                        # MSG = connected components
                        MList = getMSG(FGMod,colours)
                        for M in MList:
                            cc = set2tuple(M)
                            if cc not in temp and cc not in toAdd:
                                toAdd.add(cc)
        # toAdd - new cycles we have found. Add to todo and output temp
        todo.update(toAdd)
        temp.update(toAdd)
    return temp

def findConnection(p1,p2,F,colours,FG):
    resa = {i:checkRSGSol(F[i],colours, FG) for i in p1}
    resb = {i:checkRSGSol(F[i],colours, FG) for i in p2}
    temp = []
    for a in range(len(p1)-1):
        for b in range(len(p2)-1):
            ab = set(F[p1[a]]).union(F[p2[b]])
            resab = checkRSGSol(ab,colours,FG)
            if resab is not False and resab < (resa[p1[a]] + resb[p2[b]]):
                temp.append((a + b, a, b))
    if len(temp) == 0:
        return (0,0)
    else:
        c, a, b = max(temp)
        # print(p1,p2,a,b)
        return (a,b)

def FC(parentTree, cycles,F,colours,FG):
    '''generating set of cycles for connected component corresponding to parentTree'''
    temp = []
    for (i,j),eList in cycles.items():
        p1 = getPath(parentTree,i,False) + [i]
        p2 = getPath(parentTree,j,False) + [j]
        if len(p1) > len(p2):
            p1,p2 = p2, p1
        x = 0
        for a in range(len(p1)):
            if p1[a] == p2[a]:
                x = a
        fc = p1[x]
        fc = list(F[fc])
        p1 = p1[x:]
        p2 = p2[x:]
        a,b = findConnection(p1,p2,F,colours,FG)
        p1 = p1[a:]
        p2 = p2[b:]
        c = p1 + list(reversed(p2))
        c = [j for x in c for j in F[x]]
        temp.append(set2tuple(c))
    return temp


#####################################
## Binary Matrices for input to product constructions
#####################################

def even_rows(A):
    '''check if the rows of A are of even weight'''
    w = np.mod(np.sum(A,axis=-1),2)
    if np.sum(w) > 0:
        return False
    return True

def make_even(C):
    '''Check if each column of C is even weight. If not, add a row to ensure this'''
    myRow = np.mod(np.sum(C,axis=-0),2)
    if np.sum(myRow) > 0:
        C = np.vstack([C,[myRow]])
    return C

def BinMatRandom(m: int = 3, n: int = 4,even=True):
    '''return a random binary matrix'''
    ## Generate all binary matrices of shape (m,n)
    C = generateEvenMats(m,n) if even else generateBinMats(m,n)
    ## Choose a random index
    r = np.random.randint(len(C))
    ## Return corresponding matrix
    return C[r]

def BinMatCanonical(C):
    '''check if binary matrix C is in canonical form: both rows and cols of increasing weight'''
    ## Check rows
    w = [np.sum(c) for c in C]
    if not nonDecreasing(w):
        return False
    ## Check cols
    w = [np.sum(c) for c in C.T]
    if not nonDecreasing(w):
        return False
    return True

## Dynamic programming - global variable with previously generated matrices
BinMats = dict()

def generateBinMats(m,n):
    '''Generate a binary matrix from each equivalence class
    up to permutations of rows and cols'''
    global BinMats 
    ## if m < n, we swap m and n
    transp = m < n
    if transp:
        m,n = n,m
    ix = (m,n)
    if m == 1:
        ## exit recursion
        temp = [ZMat([[1]])]
    elif ix in BinMats:
        ## we have previously generated matrices of this shape
        temp = BinMats[ix]
    else:
        ## generate from matrices of shape (m-1,n)
        temp = []
        for C in generateBinMats(m-1,n):
            ## consider prepending a row of weight 1 <= w <= wt(C[0])
            for w in range(1,np.sum(C[0])+1):
                ## binary vectors of weight w and length n
                for r in iter.combinations(range(n),w):
                    r = set2Bin(n,r)
                    ## prepend r
                    rC = np.vstack([[r],C])
                    ## check if the result is of canonical form
                    if BinMatCanonical(rC):
                        ## if so, append the result to temp
                        temp.append(rC)
        ## update the global variable holding previous results
        BinMats[ix] = temp 
    ## if necessary, return list of transposed matrices
    if transp:
        temp = [C.T for C in temp]
    return temp

def generateEvenMats(m,n):
    MatChoices = generateBinMats(m,n)
    MatChoices = [A for A in MatChoices if even_rows(A)]
    MatChoices = [A for A in MatChoices if even_rows(A.T)]
    return MatChoices

#################################################
##  analysis of codes
#################################################

def analyseFlagCode(C,constrFun=Complex_to_PinCode,SG=False,coloured_logical_paulis=False,calc_dist=False,calc_LO=False):
    SX, SZ, LX, LZ = constrFun(C,SG=SG,CP=coloured_logical_paulis)
    print('\nWeights of stabiliser generators and logical Paulis:')
    print('SX',freqTablePrint(np.sum(SX,axis=-1)))
    print('SZ',freqTablePrint(np.sum(SZ,axis=-1)))
    print('LX',freqTablePrint(np.sum(LX,axis=-1)))
    print('LZ',freqTablePrint(np.sum(LZ,axis=-1)))
    dX = 0 if len(LX) == 0 else min(np.sum(LX,axis=-1))
    dZ = 0 if len(LX) == 0 else min(np.sum(LZ,axis=-1))
    ## dimension of complex = Clifford hierarchy level
    D = len(C)
    k,n = np.shape(LX)
    temp = []
    temp.append(f'[[{n},{k}]]')
    if len(LX) > 0:
        comm = matMul(LX,SZ.T,2)
        # print('matMul(LX,SZ.T,2)',elapsedTime())
        temp.append("LX commute" if np.sum(comm) == 0 else "err")
        comm = matMul(LZ,SX.T,2)
        # print('matMul(LZ,SX.T,2)',elapsedTime())
        temp.append("LZ commute" if np.sum(comm) == 0 else "err")
        temp.append('QTrans' if QuasiTransversal(SX,LX,D) else 'not trans')
        # print('Qtrans',elapsedTime())
        if dZ > 4 and calc_dist:
            SZLZ = np.vstack([SZ,LZ])
            SZLZ = getH(SZLZ,2)
            SZLZ = minWeight(SZLZ)
            SZLZ = ZMat(SZLZ)
            dZ = n
            for z in SZLZ:
                w = np.sum(z) 
                if w < dZ and np.sum(matMul(z,LX.T,2)) > 0:
                    dZ = w
        temp.append(f'dZ {dZ}')
        gamma = codeGamma(n,k,dZ)
        temp.append(f'gamma {gamma}')
    else:
        temp.append('No encoded qubits')
    if calc_LO:
        ## LO including CZ gates at level 3 of the Clifford hierarchy
        # CZLO(SX,LX)
        ## Transversal logical operators involving only single-qubit gates
        LOReport(SX,LX,SZ,LZ,D)
    return "\t".join(temp)


def LOReport(SX,LX,SZ,LZ,t):
    N = 1 << t
    print('\nCalculating Transversal Logical Operators')
    zList, qList, V, K_M = comm_method(SX, LX, SZ, t,compact=True,debug=False)
    print(f'(action : z-component)')
    for z, q in zip(zList,qList):
        print( CP2Str(2*q,V,N),":", z2Str(z,N))