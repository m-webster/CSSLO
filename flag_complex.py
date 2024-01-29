from common import *
from NSpace import *
from clifford_LO import *
import itertools as iter

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

def getFlags(C):
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
        k = F[j].copy()
        k[i] = 0
        k = tuple(k)
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

# def maximalSubgraphs(FG,colours):
#     '''Generate set of maximal subgraphs for the flag graph FG and set of colours/dimensions colours'''
#     n = len(FG[0])
#     d = colours[0]
#     F = [cycle2tuple(FG[d][i].union({i})) for i in range(n)]
#     if len(colours) == 1:
#         return set(F)
#     ## merge adjacency matrices
#     A = [set().union(*[FG[d][i] for d in colours]) for i in range(n)]
#     temp = {cycle2tuple(parentTree.keys()) for (parentTree, cycles) in spanningTrees(F,A)}
#     return temp 

def maximalSubgraphs(FG,colours):
    '''Generate set of maximal subgraphs for the flag graph FG and set of colours/dimensions colours'''
    todo = set(range(len(FG[0])))
    temp = []
    while len(todo) > 0:
        i = todo.pop()
        res = MSGExplore(FG,colours,i)
        temp.append(cycle2tuple(res))
        todo = todo - res
    return temp 

def MSGExplore(FG,colours,i):
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

def mergeFaces(F,FG,colours):
    '''merge partial RSGs together to make rainbow subgraphs'''
    ## max number of times a face can be used
    maxF = 1
    ## list of completed rainbow subgraphs
    temp = set()
    ## convert F to list of tuples
    F = list({cycle2tuple(c) for c in F})
    ## calculate how many flags have all colours
    scores = [checkRSGPartial(c,colours, FG) for c in F]
    ## order in increasing ratio of full colourings
    ix = argsort([scores[i]/len(F[i]) for i in range(len(scores))])
    startingPoints = []
    ## startingPoints are cycles which are valid but not closed
    for i in ix:
        ## check if cycle is valid
        if scores[i] is not False:
            ## check if cycle is closed
            if (scores[i] == len(F[i]) * len(colours)):
                ## if closed, add to result
                temp.add(F[i])
            else:
                ## otherwise, add to startingpoints
                startingPoints.append(F[i])
    ## iterate through startingpoints
    faceCount = {c:0 for c in startingPoints}
    while len(startingPoints) > 0:
        c = startingPoints.pop()
        ## faces included in RSG
        RSGfaces = {c}
        ## flags included in RSG
        RSG = set(c)
        lastScore = 0
        status = 0
        while status == 0:
            ## cycles which can be validly merged to form the rainbow subgraph
            validCycles = []
            for d in faceCount.keys():
                if d not in RSGfaces and faceCount[d] <= maxF:
                    cd = RSG.union(d)
                    ## check how many flags have all colours; False if there are conflicts
                    res = checkRSGPartial(cd,colours, FG)
                    if res is not False:
                        ## res/len(cd) - proportion of flags with all colours
                        validCycles.append((res/len(cd),res,d))
            ## if validCycles == [], all remaining cycles conflict
            if (len(validCycles) == 0):
                status = 2
            if status == 0:
                ## find the best cycle to merge
                score,  res,  d = max(validCycles)
                ## merge d into RSG
                RSG.update(d)
                if lastScore > score:
                    ## we are getting worse - abandon this pathway
                    status = 2
                else: 
                    RSGfaces.add(d)
                    lastScore = score
                    ## if res == len(RSG), then we have all colours at each flag of RSG
                    if (res == len(RSG) * len(colours)):
                        status = 1
                        ## update faceCount and startingPoints for faces in RSGfaces or which are a subset of RSG
                        for d in faceCount.keys():
                            if d in RSGfaces or set(d).issubset(RSG):
                                faceCount[d] += 1
        ## double-check if this is a valid RSG and if so add to temp
        if status == 1:
            temp.add(cycle2tuple(RSG))
    return temp

def rainbowSubgraphs(FG,colours):
    if len(colours) == 1:
        d = colours[0]
        return {cycle2tuple(FG[d][i].union({i})) for i in range(len(FG[d]))}
    temp = set()
    F = rainbowSubgraphs(FG,colours[1:])
    d = colours[0]
    FC = faceCycles(F,FG[d],colours)
    if len(colours) > 2:
        temp.update(mergeFaces(FC,FG,colours)) 
    else:
        return FC
    return temp 

def faceCycles(F,A,colours):
    F = list(F)
    temp = set()
    for parentTree, cycles in spanningTrees(F,A):
        # print('parentTree')
        # for k,p in parentTree.items():
        #     if p is not None:
        #         p,e = p
        #         print(F[p],"=",e,"=>",F[k])
        # print('cycles')
        # for (a,b),eList in cycles.items():
        #     for e in eList:
        #         print(F[a],"=",e,"=>",F[b]) 
        if len(colours) == 2:
            CC = inducedCycles(parentTree, cycles)
        else:
            CC = FC(parentTree,cycles,F)
        temp.update({cycle2tuple(c) for c in CC})
    return temp

def spanningTrees(F,A):
    '''Return spanning tree and unvisited edges for each connected component of flag graph with ordered colours ix'''
    AVF = [{i for i in range(len(F)) if j in F[i]} for j in range(len(A))]
    faces2visit = set(range(len(F)))
    trees = []
    while len(faces2visit) > 0:
        ## next flag to visit
        i = faces2visit.pop()
        ## get parentTree and unvisited edges
        parentTree, cycles = ST(F,A,AVF,i)
        ## update flags still to visit
        faces2visit.difference_update(parentTree.keys())
        trees.append((parentTree, cycles))
    return trees

def nextFaces(F,A,AVF,f1,visitedFlags):
    # facesFound = set()
    # print(func_name(),'f1',f1)
    for v1 in F[f1]:
        # print('v1',v1)
        if v1 not in visitedFlags:
            for v2 in A[v1]:
                # print('v2',v2)
                if v2 not in visitedFlags:
                    for f2 in AVF[v2]:
                        # print('f2',f2)
                        # if f2 not in facesFound:
                        #     facesFound.add(f2)
                        visitedFlags.add(v1)
                        visitedFlags.add(v2)
                        yield (f2,(v1,v2))

def ST(F,A,AVF,f1):
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
        # visitedFlags = set(getPath(parentTree,f1))
        for (f2,e) in nextFaces(F,A,AVF,f1,visitedFlags):
            if (f2 not in parentTree):
                ## j not previously visited - update parentTree and todo
                parentTree[f2] = (f1,e)
                todo.append(f2)
            else:
                cix = cycle2tuple((f1,f2))
                if (cix not in cycles):
                    cycles[cix] = set()
                ## otherwise, add to cycles
                cycles[cix].add(cycle2tuple(e))
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
    # print(func_name(),p1,p2,"=>",temp)
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


def FC(parentTree, cycles,F):
    '''generating set of cycles for connected component corresponding to parentTree'''
    temp = []
    for (i,j),eList in cycles.items():
        p1 = getPath(parentTree,i,False)
        p2 = getPath(parentTree,j,False)
        c = {i,j}.union(path2cycle(p1,p2))
        # print('c',c)
        c = set().union(*[F[i] for i in c])
        # print('len(c)',len(c))
        temp.append(c)
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
    return temp

def checkRSGNodePartial(i,c,colours,FG):
    score = 0
    for d in colours:
        ## check if how many neighbours i has in c of type d
        ## there needs to be exactly one
        s = len(FG[d][i].intersection(c))
        if s > 1:
            return False
        score += s
    return score
    return 1 if score == len(colours) else 0

def checkRSGPartial(c,colours,FG):
    score = 0
    for i in c:
        res = checkRSGNodePartial(i,c,colours,FG)
        if res is False:
            return False
        score += res
    return score

def checkRSGNode(i,c,colours,FG):
    temp = []
    for d in colours:
        ## check if how many neighbours i has in c of type d
        ## there needs to be exactly one
        s = len(FG[d][i].intersection(c))
        if s != 1:
            # temp.append(f'{i}:{d}:{s}')
            temp.append(i)
    return temp

def checkRSG(c,colours,FG):
    temp = []
    for i in c:
        temp.extend(checkRSGNode(i,c,colours,FG))
    return temp

def simplifyGens(CC):
    '''simplify generators for flag code'''
    n = max(set().union(*CC)) + 1
    ## convert to binary matrix
    SXCC = [set2Bin(n,c) for c in CC]
    ## RREF
    H, rowops = How(SXCC,2)
    ## if len(SXCC) > len(H) then there are linear dependencies
    if len(SXCC) > len(H):
        ## get min weight set of generators and remove zero rows
        SXCC = minWeight(SXCC)
        SXCC = RemoveZeroRows(SXCC)
        ## independent set of generators, ordered by increasing weight
        SXCC = indepLZ(ZMatZeros((0,n)),SXCC)
    return [bin2Set(c) for c in SXCC]

def doubleComplex(C):
    C[0] = np.vstack([C[0],C[0]])
    return C

#####################################33

def printAdj(A):
    for i in range(len(A)):
        print(i,":",A[i])
    
def cycle2tuple(c):
    return tuple(sorted(c))

def CC2n(CC):
    '''find n from a set of cycles CC'''
    allFlags = set().union(*CC)
    return max(allFlags) + 1 if len(allFlags) > 0 else 0

def transliterate(s,f):
    return {f[i] for i in s}

def mergeQubits(CC,FI):
    return {cycle2tuple(transliterate(s,FI)) for s in CC}

def flagIdentify(FG,ix):
    MSG = list(maximalSubgraphs(FG,ix))
    n = len(FG[0])
    temp = [-1 for i in range(n)]
    for i in range(len(MSG)):
        for s in MSG[i]:
            temp[s] = i
    return temp

def flag_code(C,Sxmc,SG=False,IdC=None):
    '''Return Flag code in binary format
    C: a list of boundary operators representing a complex
    Sxmc: list of stabiliser types encoded as tuples (x,m,c): 
    - x is either 'x' or 'z' indicating whether this is an X or Z-stabiliser
    - m is either 'r' or 'm' indicating rainbow or maximal subgraph
    - c is a list of colours for the subgraphs
    SG: if True, return a min weight set of generators of each cycle type - this can be slow
    IdC: identify qubits within specified maxmimal subgraphs- eg (0,1) identifies qubits in the same maximal (0,1) subgraph'''
    ## generate flags from complex C
    F = getFlags(C)
    ## construct Flag Graph - results in a list of adjacency dictionaries - one for each colour
    FG = flagGraph(F)
    ## initialise X and Z stabiliser lists
    SX = []
    SZ = []
    ## FI is a lookup table for identifying qubits in maximal subgraphs
    FI = None if IdC is None else flagIdentify(FG,IdC)
    for x,m,colours in Sxmc:
        CC = rainbowSubgraphs(FG,colours) if m=="r" else maximalSubgraphs(FG,colours) 
        ## optionally make min weight set of cycles
        if SG:
            CC = simplifyGens(CC)
        ## optionally identify qubits as specified by FI
        if FI is not None:
            CC = mergeQubits(CC, FI)

        print('freqTable',x,m,colours,freqTable([len(s) for s in CC]))
        if x=='x':
            SX.extend(CC)
        else:
            SZ.extend(CC)
    n = CC2n(SX + SZ)
    SX = ZMat([set2Bin(n, s) for s in SX])
    SZ = ZMat([set2Bin(n, s) for s in SZ])
    return SX,SZ

def productComplex(HList,double=False):
    d = len(HList)
    C = complexNew([HList[0]])
    for i in range(1,d):
        Ci = complexNew([HList[i]])
        C = complexTCProd(C, Ci)
    C = [c.T for c in reversed(C)]
    return C if not double else doubleComplex(C)

def Product_24_Cell(HList,double=False,SG=False):
    C = productComplex(HList[:3],double)
    IdC = [0,3]
    xCycles = [(0,1,2),(1,2,3)]
    Sxmc = [('x','m',c) for c in xCycles]
    zCycles = [(0,1),(1,2),(2,3)]
    Sxmc += [('z','r',c) for c in zCycles]
    return flag_code(C,Sxmc,SG,IdC)

def Product_8_24_48_Cell(HList,double=False,SG=False):
    C = productComplex(HList[:3],double)
    IdC = [3]
    xCycles = [(0,1,2),(0,2,3),(1,2,3)]
    Sxmc = [('x','m',c) for c in xCycles]
    zCycles = [(0,1),(0,2),(1,2),(0,1,3),(2,3)]
    Sxmc += [('z','r',c) for c in zCycles]
    return flag_code(C,Sxmc,SG,IdC)

def ProductDefault(HList,double=False,SG=False):
    C = productComplex(HList,double)
    IdC = None
    xCycles = iter.combinations(range(d+1),d)
    Sxmc = [('x','m',c) for c in xCycles]
    zCycles = iter.combinations(range(d+1),2)
    Sxmc += [('z','r',c) for c in zCycles]
    return flag_code(C,Sxmc,SG,IdC)

def Default2D(HList,double=False,SG=False):
    C = productComplex(HList[:2],double)
    IdC = None
    rCycles = [(0,1),(1,2)]
    mCycles = [(0,2)]
    Sxmc = [('x','m',c) for c in mCycles] + [('x','r',c) for c in rCycles]
    Sxmc += [('z','m',c) for c in mCycles] + [('z','r',c) for c in rCycles]
    return flag_code(C,Sxmc,SG,IdC)


def random_code(n_rows: int = 3, n_cols: int = 4):
    matrix = []
    while len(matrix) < n_cols:
        col = np.random.choice([0, 1], n_rows)
        if np.sum(col) % 2 == 0:
            matrix.append(col)
    return np.transpose(matrix)
	
def make_even(H: np.ndarray) -> np.ndarray:
    """Take a (classical) parity-check matrix and make its rows and columns even-weight
    by potentially adding a new row and a new column.

    Parameters
    ----------
    H : np.ndarray
       Parity-check matrix

    Returns
    -------
    np.ndarray
        Even-weight parity-check matrix
    """
    new_col = np.array([np.sum(H, axis=1) % 2]).T
    if not np.all(new_col == 0):
        H = np.hstack([H, new_col])

    new_row = np.sum(H, axis=0) % 2
    if not np.all(new_row == 0):
        H = np.vstack([H, new_row])

    return H

	
#################################################

## make toric code of dimension d and cycle length L
d = 3
L = 2
H = repCode(L)
HList = [H for i in range(d)]
## add random classical codes to product
# HList[0] = make_even(random_code(3,4))
# HList[1] = make_even(random_code(5,6))

print('Classical Codes')
for i in range(len(HList)):
    print(f'H_{i}')
    print(ZmatPrint(HList[i]))

## SG=True - return min weight stabilisers - takes longer
SG = True

## default product code type
SX, SZ = ProductDefault(HList,double=False,SG=SG)

## default 2D code
# SX, SZ = Default2D(HList,double=False,SG=True)

## These are all 3D tyes requiring d=3

## Based on 8-24-48 honeycomb
# SX, SZ = Product_8_24_48_Cell(HList,double=False)

## Based on 24-cell honeycomb
# SX, SZ = Product_24_Cell(HList,double=False)

## basic analysis of code
H, rowops = How(SX,2)
## rank of SX
print('SX Rank',len(H))
## check if code is d-orthogonal
print('SX tOrthogonal',tOrthogonal(SX,d))

myCode = CSSCode(SX,SZ=SZ)
Eq,SX,LX,SZ,LZ = myCode
k,n = np.shape(LX) 
## code parameters
print(f'[[{n},{k}]] Code')

## calculate distance - can be slow
if len(LZ) > 0:
    LZ = minWeightLZ(SZ,LZ)
    d = min(np.sum(LZ,axis=-1))
    print('Z-Distance',d)
