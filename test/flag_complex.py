from add_parent_dir import *
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

def maximalSubgraphs(FG,colours):
    '''Generate set of maximal subgraphs for the flag graph FG and set of colours/dimensions colours'''
    temp = []
    ## there is a parentTree for connected component
    for parentTree, cycles in spanningTrees(FG,colours):
        temp.update(set(parentTree.keys()))
    return temp

def rainbowSubgraphs(FG,colours):
    '''Generate set of rainbow subgraphs for the flag graph FG and set of colours/dimensions colours
    optimisation based on spanning trees - currently only works for colour sets of size 2'''
    temp = []
    ## go through each permutation of the colours
    for ix in iter.permutations(range(len(colours))):
        ## only consider permutations such that 0 appears before 1 - avoids double-counting of edges
        if incPerm(ix):
            ix = [colours[i] for i in ix]
            ## create spanning tree and edges which are not visited - these correspond to a generating set of cycles
            for parentTree, cycles in spanningTrees(FG,ix):
                ## convert to proper cycles
                for c in inducedCycles(parentTree, cycles):
                    temp.append(c)
    '''Handling 3+ colours idea: 3 colour rainbow subgraphs corresponds to polytopes
    1. identify (0,1) cycles - correspond to faces
    2. identify (0,1,2) cycles - correspond to loops around polytopes
    3. link overlapping faces to loops to form partial polytopes - repeat till polytopes are closed
    4. iterate for higher numbers of colours - eg (0,1,2,3) cycles linking 3D polytopes'''
    return temp

def incPerm(ix):
    '''in the permutation ix is 0 to the left of 1'''
    # return True
    return ix.index(0) < ix.index(1)

def spanningTrees(FG,ix):
    '''Return spanning tree and unvisited edges for each connected component of flag graph with ordered colours ix'''
    flagsTovisit = set(range(len(FG[0])))
    trees = []
    while len(flagsTovisit) > 0:
        ## next flag to visit
        i = flagsTovisit.pop()
        ## get parentTree and unvisited edges
        parentTree, cycles = ST(FG,ix,i)
        ## update flags still to visit
        flagsTovisit.difference_update(parentTree.keys())
        trees.append((parentTree, cycles))
    return trees


def NextFlag(i,s,ix,FG):
    '''iterator for flags to visit next
    FG: flag graph FG
    ix: ordered colours ix
    s: remaining colours to be visited
    i: index of current flag'''
    ## at the end of colour iteration - reset s
    if len(s) == 0:
        s = ix
    d = s[0]
    ## iterate through each flag adjacent to FG[d][i]
    for j in FG[d][i]:
        yield (j, s[1:])

def ST(FG,ix,i):
    '''Generate spanning tree and unvisited edges corresponding to cycles
    FG: flag graph
    ix: ordered colours
    i: index of starting flag'''
    cycles = set()
    todo = [(i,[])]
    parentTree = {i:None}
    while len(todo) > 0:
        ## i - flag index; s - sequence of colours to visit
        i,s = todo.pop()
        for j,t  in NextFlag(i,s,ix,FG):
            if (j not in parentTree):
                ## j not previously visited - update parentTree and todo
                parentTree[j] = i
                todo.append((j,t))
            else:
                ## otherwise, add to cycles
                cycles.add((i,j))
    return parentTree,cycles

def path2cycle(p1,p2):
    '''form a cycle from path up to i, joined to p[-1] via the path p'''   
    n = min(len(p1),len(p2))
    ## find first place where p1 and p2 vary, going from L to R (avoid subcycles)
    i = 0
    while i < n and p1[i] == p2[i]:
        i+=1
    f = i - 1
    ## join the paths to form a cycle
    temp = p1[f:]  + list(reversed(p2[f+1:]))
    ## check if the cycle is even length
    ## should be OK because for flags a, b, c: a ~ c via vertex and b ~ c via vertex => a ~ b via vertex as a,b,c in same edge
    if (len(temp) % 2 == 1):
        return temp[1:]
    return temp


def getPath(parentTree,i):
    '''Return path of flags visited to reach flag i'''
    temp = [i]
    while parentTree[i] is not None:
        j = parentTree[i]
        temp.append(j) 
        i = j
    return list(reversed(temp))

def inducedCycles(parentTree, cycles):
    '''generating set of cycles for connected component corresponding to parentTree'''
    temp = []
    for (i,j) in cycles:
        c = path2cycle(getPath(parentTree,i),getPath(parentTree,j))
        temp.append(c)
    return temp

#####################################33

def adjMerge(AList):
    return [set().union(*[A[i] for A in AList]) for i in range(len(AList[0]))]

def printAdj(A):
    for i in range(len(A)):
        print(i,":",A[i])

    
def cycle2tuple(c):
    return tuple(sorted(c))
    i = np.argmin(c)
    print(func_name(),c)
    c = tuple(np.roll(c,-i))

def transliterate(s,f):
    return {f[i] for i in s}

def mergeQubits(SX,FI):
    return [transliterate(s,FI) for s in SX]

def flagIdentify(FG,ix):
    A = []
    n = len(FG[0])
    for i in ix:
        for j in range(n):
            A.append({j}.union(FG[i][j]))
    A = mergeAdj(A)
    temp = [-1 for i in range(n)]
    for i in range(len(A)):
        for s in A[i]:
            temp[s] = i
    return temp

#######################################3

## toric codes
d = 3
L = 2

H = repCode(L)
C = complexNew([repCode(L)])
C1 = complexNew([H])
for i in range(d-1):
    C = complexTCProd(C, C1)
# double complex
    
C[0] = np.vstack([C[0],C[0]])
# for i in range(len(C)):
#     print(f'D{i}')
#     print((ZmatPrint(C[i])))

## Manual entry of Complex

# # Doublecomplex (3,3) - Rank 97
# C0 = '''101000000100000100
# 101000000100000100
# 110000000010000010
# 110000000010000010
# 011000000001000001
# 011000000001000001
# 000101000100100000
# 000101000100100000
# 000110000010010000
# 000110000010010000
# 000011000001001000
# 000011000001001000
# 000000101000100100
# 000000101000100100
# 000000110000010010
# 000000110000010010
# 000000011000001001
# 000000011000001001'''
# C1 = '''100000100
# 010000010
# 001000001
# 100100000
# 010010000
# 001001000
# 000100100
# 000010010
# 000001001
# 101000000
# 110000000
# 011000000
# 000101000
# 000110000
# 000011000
# 000000101
# 000000110
# 000000011'''

# ##############################################
# ## Random Complex Rank 50

# C0 = '''100100000101000000000000
# 010010000110000000000000
# 001001000011000000000000
# 000000000000101000000000
# 000000000000110000000000
# 000000000000011000000000
# 000100100000000101000000
# 000010010000000110000000
# 000001001000000011000000
# 100100000000000000101000
# 010010000000000000110000
# 001001000000000000011000
# 000100100000000000000101
# 000010010000000000000110
# 000001001000000000000011'''
# C1 = '''101000000
# 110000000
# 011000000
# 000101000
# 000110000
# 000011000
# 000000101
# 000000110
# 000000011
# 100100000
# 010010000
# 001001000
# 000000000
# 000000000
# 000000000
# 000100100
# 000010010
# 000001001
# 100100000
# 010010000
# 001001000
# 000100100
# 000010010
# 000001001'''

# ###################################
# ## 3D Colour Code
# ###################################
# C = []
# C.append('''100010001010000011000000
# 010001000101000011000000
# 001000101010000000110000
# 000100010101000000110000
# 100010000000101000001100
# 010001000000010100001100
# 001000100000101000000011
# 000100010000010100000011''')
# C.append('''101000001100000000000000
# 010100001100000000000000
# 101000000011000000000000
# 010100000011000000000000
# 000010100000110000000000
# 000001010000110000000000
# 000010100000001100000000
# 000001010000001100000000
# 100010000000000011000000
# 010001000000000011000000
# 001000100000000000110000
# 000100010000000000110000
# 100010000000000000001100
# 010001000000000000001100
# 001000100000000000000011
# 000100010000000000000011
# 000000001000100010100000
# 000000000100010001010000
# 000000000010001010100000
# 000000000001000101010000
# 000000001000100000001010
# 000000000100010000000101
# 000000000010001000001010
# 000000000001000100000101''')
# C.append('''11000000
# 11000000
# 00110000
# 00110000
# 00001100
# 00001100
# 00000011
# 00000011
# 10100000
# 01010000
# 10100000
# 01010000
# 00001010
# 00000101
# 00001010
# 00000101
# 10001000
# 01000100
# 00100010
# 00010001
# 10001000
# 01000100
# 00100010
# 00010001''')

# C = [bin2Zmat(c) for c in C]

##############################################

d = len(C)
F = getFlags(C)
n = len(F)
print(f'n={n}')

FG = flagGraph(F)
sep = " " if n > 10 else ""
FStr = [sep.join([str(a) for a in f]) for f in F]
SX = []


for ix in iter.combinations(range(d+1),d):
    # CC = maxSubgraphs(FG,ix)
    CC = rainbowSubgraphs(FG,ix)
    SX.extend(CC)
    print('Rainbow Subgraphs',ix)
    print(freqTable([len(s) for s in CC]))
    for c in CC:
        print(", ".join([FStr[i] for i in c]))
    SXCC = [set2Bin(n,c) for c in CC]
    H, rowops = How(SXCC,2)
    print('rank',len(H))
    # printAdj(CC)
    # CC = maxSubgraphs(FG,ix)
    # print('Maximal Subgraphs',ix)
    # print(freqTable([len(s) for s in CC]))
    # printAdj(CC)
    # CC = mergeAdj(CC)
    # print('Maximal colourGraph',ix)
    # print(freqTable([len(s) for s in CC]))
SX = [set2Bin(n, s) for s in SX]
H, rowops = How(SX,2)
print(len(H))

# print(freqTable([len(s) for s in SX]))
exit()
FI = flagIdentify(FG,[3])
SX = mergeQubits(SX, FI)
print(freqTable([len(s) for s in SX]))