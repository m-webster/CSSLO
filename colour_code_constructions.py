# from add_parent_dir import *
from common import *
from NSpace import *
from clifford_LO import *
import itertools as iter

##########################################################
## Reports
##########################################################

def bicolouringAnalysis(Cells,t):
    E,F = Cells[0],Cells[1]
    SX = Cells[-1]
    SZ = Cells[1]
    N = 1 << t
    ## check if SX/SZ rep communting operators
    comm = matMul(SX,SZ.T,2)
    print("\n")
    if np.sum(comm) == 0:
        ## make a CSS code
        Eq,SX,LX,SZ,LZ = CSSCode(SX,SZ=SZ)
        k,n = np.shape(LX)
        ## Code parameters
        print(f'[[{n},{k}]] Code')

        ## make min weight set of LX/LZ - optional
        # LX = minWeightLZ(SX,LX)
        # LZ = minWeightLZ(SZ,LZ)

        if k > 0:
            ## vertex bi-colouring - set form
            B = VColouring(E)    
            ## convert to binary vector      
            B = set2Bin(n,B)
            ## testing only...
            # Cycles = EF2Cycles(Cells[0],Cells[1])
            # B = biColouring(Cycles)
            ## check if a valid bicolouring
            if len(B) > 0 and max(B) < 2:                
                print('biColouring',bin2Set(B))
                ## Make Logical operator from Bicolouring
                L = np.mod((-1) ** B,N)
                ## phase applied to codewords
                res = checkDiagLO(L,Eq,SX,LX,SZ,t)
                if res is False:
                    print('Not a Logical Operator')
                else:
                    q,V = res
                    print('Logical operator with action',CP2Str(2*q,V,N))
            else:
                print('No Bicolouring')
                LOReport(Eq,SX,LX,SZ,LZ,t)
    else:
        print('Non-commuting stabilisers')

def checkDiagLO(L,Eq,SX,LX,SZ,t):
    N = 1 << t
    # calculate logical identities
    if t > 2:
        # for t>2, we need to do this explicitly
        K_M = getKM(Eq,SX,LX,N//2) * 2
    else:
        # otherwise, use the Z-checks
        # ensure they are in RREF
        SZ, rowps = How(SZ,2)
        r,n = np.shape(SZ)
        ## append a column of zeros for phase
        K_M = np.hstack([SZ,ZMatZeros((r,1))]) * (N//2)
    if not isDiagLO(L,SX,K_M,N):
        return False
    pList, V = DiagLOActions([L],LX,N)
    # isDiagLO(B,SX,K_M,N)
    ## actions as CP operators
    q = action2CP(V,pList[0],N)
    return q, V

def LOReport(Eq,SX,LX,SZ,LZ,t):
    N = 1 << t
    print('\nCalculating Transversal Logical Operators')
    zList, qList, V, K_M = comm_method(Eq, SX, LX, SZ, t,compact=True,debug=False)

    print(f'(action : z-component)')
    for z, q in zip(zList,qList):
        print( CP2Str(2*q,V,N),":", z2Str(z,N))
    
def codeStats(Cells,calcDist=True, calck=True,LO=False):
    SX = Cells[-1]
    SZ = Cells[1]
    nSX,n = np.shape(SX)
    nSZ,n = np.shape(SZ)
    
    wSX = freqTable(np.sum(SX,axis=-1))
    wSZ = freqTable(np.sum(SZ,axis=-1))
    k, dZ,wLZ,gamma = 'NA','NA','NA','NA'
    comm = np.sum(matMul(SX,SZ.T,2))
    if comm == 0 and calck:
        Eq, SX,LX,SZ,LZ = CSSCode(SX,SZ=SZ)
        k,n = np.shape(LX)
        # print(func_name(), f'n={n}')
        # E = getCells(SZ)
        # print('E',E)
        if k > 0 and calcDist:
            startTimer()
            LZ = minWeightLZ(SZ,LZ)
            wLZ = freqTable(np.sum(LZ,axis=-1))
            # print('wLZ',wLZ)
            dZ = np.sum(LZ[0])
            # print("Z-distance Method 1",elapsedTime())
            # dZ1 = GRANDdX(SX,LX,dZ)
            # print("Z-distance GRAND",elapsedTime())
            # print(f'dZ={dZ}; GRAND method dZ={dZ1}')

            # LX = minWeightLZ(SX,LX)
            # dX = np.sum(LX[0])
            dX = 'NA' 

            # d = min(dX,dZ)
            d = dZ
            gamma = codeGamma(n,k,d)
            gamma = f'{gamma:.2f}'
        if LO is not False:
            LX = minWeightLZ(SX,LX)
            LOReport(Eq,SX,LX,SZ,LZ,LO)
    return f'{n}\t{k}\t{dZ}\t{gamma}\t{nSX}\t{nSZ}\t{wLZ}\t{wSX}\t{wSZ}'



##########################################################
## Reflection Group Constructions
##########################################################

def C2E(Cells,d,w=None):
    '''find intersections of Cells. d sets how many cells to take the intersection of. By setting w=2, we find all edges'''
    E = set()
    ## sets of Cells of size d
    for ix in iter.combinations(range(len(Cells)),d):
        e = np.product(Cells[ix,:],axis=0)
        ## ew is weight of intersection
        ew = np.sum(e)
        ## here we select only intersections of desired weight w
        if (w is None and ew > 0) or (ew == w):
            E.add(tuple(e))
    return ZMat([e for e in E])


def RGCellulation(myrow,d):
    '''Make a  colour code based on reflection group cellulation.'''
    if d ==2:
        F = str2ZMatdelim( myrow['zFV'])
        E = str2ZMatdelim( myrow['zEV']).T
        return [E,F]
    else:
        C = str2ZMatdelim( myrow['zSX'])
        F = str2ZMatdelim( myrow['zSZ'])
        E = C2E(F,2,w=2)
        return [E,F,C]
    
def SXCellulation(SX,d):
    '''Cellulation based on X-checks SX corresponding to highest-dimensional cells'''
    if d==2:
        ## Faces are SX
        F = SX
        ## Edges - interseection of sets of faces of size 2 with weight 2
        E = C2E(SX,2,w=2)
        Cells = [E,F]
    if d==3:
        ## 3-dimensional cells are SX
        C = SX
        ## intersections of 2 cells are faces
        F = C2E(SX,2)
        ## Edges - interseection of sets of cells of size 3 with weight 2
        E = C2E(SX,3,w=2)
        r,n = np.shape(SX)
        ## calculate cycles to split out faces - eg where cells intersect twice for smaller codes
        Cycles = EF2Cycles(Cells[0],Cells[1])
        Cells[1] = ZMat([set2Bin(n,c) for c in Cycles])
        Cells = [E,F,C]
    return Cells

def EF2Cycles(E,F):
    '''Find cycles from Edges and Faces - these are a sequence of vertices where the next vertex is joined by an edge '''
    Cycles = []
    for f in F:
        EList = []
        for e in E:
            if np.sum(e * f) == 2:
                EList.append(bin2Set(e))
        Cycles.extend(getCycles(EList))
    return Cycles


def getCycles(EList):
    '''Split vertices into sets joined by edges'''
    FList = []
    ## adjacency dict
    Vadj = dict()
    for e in EList:
        for i in range(2):
            if e[i] not in Vadj:
                Vadj[e[i]] = []
            Vadj[e[i]].append(e[i-1])
    ## vertices already explored
    explored = set()
    ## vertices to explore
    toExplore = []
    while len(Vadj) > 0:
        ## nothing to explore - find a new starting point
        if len(toExplore) == 0:
            ## new face
            F = []
            v = min(Vadj.keys())
            ## add v to explored and toExplore
            explored.add(v)
            toExplore.append(v)
        while len(toExplore) > 0:
            ## next vertex
            v = toExplore.pop()
            F.append(v)
            ## explore adjacent vertices
            for u in Vadj[v]:
                if u not in explored:
                    explored.add(u)
                    toExplore.append(u)
            ## delete from adj Dict
            del Vadj[v]
        ## Face completed - add to FList
        FList.append(F)
    return FList

#############################
## Colourings of Cellulations
#############################

def checkColouring(Adj,ix):
    '''Check if a colouring ix for adjacency matrix Adj is valid - i.e. neighbours of ix and ix do not overlap'''
    n = len(Adj)
    for i in range(n):
        Adj[i,i] = 0
    ## neighbours of ix
    NN = np.sum(Adj[ix],axis=0)
    ## convert to binary vector
    ix = set2Bin(n,ix)
    ## check that ix and NN have no overlap
    return np.sum(ix * NN) == 0

def VColouring(E):
    '''return bicolouring of Vertices'''
    m,n = np.shape(E)
    Adj = ZMatZeros((n,n))
    for e in E:
        ## handle hyperedges
        e = bin2Set(e)
        for (i,j) in iter.combinations(e,2):
            Adj[i,j] = 1
            Adj[j,i] = 1
    ix = multiComponentColouring(Adj) 
    print(func_name(),'valid colouring', checkColouring(Adj,ix))
    return ix

def FColouring(F):
    '''return set of non-intersecting faces'''
    r = len(F)
    Adj = ZMatZeros((r,r))
    for i in range(r):
        for j in range(r):
            ## check if they share a vertex
            if np.sum(F[i] * F[j]) > 1:
                Adj[i,j] = 1
    ix = multiComponentColouring(Adj) 
    print(func_name(),'valid colouring', checkColouring(Adj,ix))
    return F[ix]

def multiComponentColouring(Adj):
    '''Colouring where we are not guaranteed that there is a single connected component'''
    temp = []
    r = len(Adj)
    ## add diagonal entries
    for i in range(len(Adj)):
        Adj[i,i] = 1
    while not np.sum(Adj) == 0:
        ## find elts of same colour
        ix = getColouring(Adj)
        temp.extend(ix)
        ## elimate entries either in ix or neighbouring ix
        w = np.sum(Adj[ix], axis=0)
        w += set2Bin(r,ix)
        w = ZMat([0 if s > 0 else 1 for s in w])

        Adj = Adj * w 
        Adj = Adj.T * w
    return temp

def getColouring(Adj):
    '''Input is adjacency matrix Return indices of elements distance 2 apart'''
    ## square adjacency matrix to find neighbours of neighbours
    Adj2 = matMul(Adj,Adj,2)
    ## find non-zero row of Adj
    w = np.sum(Adj,axis=0)
    i = 0
    while w[i] == 0:
        i += 1
    ## initialise Depth-first search
    todo = [i]
    visited = {i}
    while len(todo) > 0:
        i = todo.pop()
        ## neighbours of neighbours
        for j in bin2Set(Adj2[i]):
            if j not in visited: 
                todo.append(j)
                visited.add(j)
    return list(visited)

           
##########################################################
## Graph Product Constructions
##########################################################

## Make 3D and 2D Graph Products

def graphProduct3D(EList,TCells):
    C = makeCellulation(TCells,EList)
    # F = GetFaces(C)
    C,V = translate_V(C)
    # F = [[V[k] for k in Fij] for Fij in F]
    n = len(V)
    CV = list(C.values())
    SX = ZMat([set2Bin(n,c) for c in CV])
    # SZ = ZMat([set2Bin(n,c) for c in F])
    return SX

def graphProduct2D(EList,TCells):
    C = makeCellulation(TCells,EList)
    C,V = translate_V(C)
    n = len(V)
    CV = list(C.values())
    SX = ZMat([set2Bin(n,c) for c in CV])
    return SX

def makeCellulation(TCells,EList):
    ## number of edges for each element of EList
    nList = [len(E) for E in EList]
    EList = [addEix(E) for E in EList]
    E = dict()
    for Eix in iter.product(*[range(n) for n in nList]):
        ## Form of eList: (v0,v1,e + len(V))
        eList = [EList[i][Eix[i]] for i in range(len(Eix))]
        for EVix,vList in TCells.items():
            EVix = translateEVix(EVix,eList)
            vList = [translatev(v,eList) for v in vList]
            append_vList(E,EVix,vList)
    return E

## Constructor functions for Cellulations

def translate_V(E):
    V = graphVertices(E.values())
    Vdict = {V[i]: i for i in range(len(V))}
    return {k:{Vdict[v] for v in e} for k,e in E.items()},Vdict


def dictSumm(D):
    for k,v in D.items():
        print(k,":",len(v))

def graphVertices(E):
    V = set()
    for e in E:
        V.update(e)
    return sorted(V)    

def axis2EVix(a,z,x,y):
    EVix = [2]*3
    EVix[a] = z 
    return tuple(EVix + [x,y])

def translateEVix(EVix,eList):
    return tuple(eList[i][EVix[i]] for i in range(len(EVix)))

def translatev(v,eList):
    d = len(eList)
    EVix, xy = v[:d], v[d:]
    return tuple(np.hstack([translateEVix(EVix,eList), xy ]))

def append_vList(E,eix,vList):
    if eix not in E:
        E[eix] = set()
    E[eix].update(vList)

def addEix(E):
    ## add edge number to end of each edge in E
    n = len(graphVertices(E))
    temp = []
    for i in range(len(E)):
        e = list(E[i])
        e.append(n + i)
        temp.append(e)
    return temp

## Cell Templates

def TemplateHex2D():
    '''3-valent cellulation using Hexagons for 2D colour code'''
    VList = [(2,2,0),(2,2,1)]
    Cells = {(0,0) : [VList[0]], (1,1): [VList[1]], (0,1) : VList, (1,0): VList}
    return Cells

def TemplateSquare2D():
    '''3-valent cellulation using Squares + Octagons for 2D colour code'''
    VOrder = [(0,0),(0,1),(1,1),(1,0)]
    nV = len(VOrder)
    Cells = {(2,2):[tuple([2,2] + list(v)) for v in VOrder]}
    VList = [tuple([2,2] + list(ix)) for ix in VOrder]
    for i in range(nV):
        ix = VOrder[i]
        Cells[ix] = [VList[i],VList[(i+1) % nV]]
    return Cells

def Template_12_24_48_Cell():
    FList = [(0,0),(0,2),(0,1),(2,1),(1,1),(1,2),(1,0),(2,0)]
    n = len(FList)
    VList = [FList[2*i]+(z,) for i in range(n//2) for z in range(2) ]
    temp = dict()
    eee = (2,2,2)
    for a in range(3):
        for z in range(2):
            vee = tuple(np.roll([2,2,z],a))
            for i in range(n):
                k = tuple(np.roll(FList[i]+(z,),a))
                v = vee + VList[i]
                # v = vee + tuple(np.roll(VList[i],a))
                dictListAdd(temp,eee,v)
                dictListAdd(temp,k,v)
                v = vee + VList[(i+1) % n]
                # v = vee + tuple(np.roll(VList[(i+1) % n],a))
                dictListAdd(temp,eee,v)
                dictListAdd(temp,k,v)
    return temp

def dictListAdd(D,k,v):
    '''add entry k->v to dictionary D'''
    if k not in D:
        D[k] = []
    D[k].append(v)

def Template24Cell():
    '''4-valent cellulation using 24 Vertex Truncated Octahedrons for 3D colour code'''
    VOrder = [(0,0),(0,1),(1,1),(1,0)]
    nV = len(VOrder)
    nextV = {VOrder[i] : VOrder[(i+1) % nV] for i in range(nV)}
    Cells = dict()
    for k in iter.product(range(2),repeat=3):
        f = []
        for a in range(3):
            (z,x,y) = np.roll(k,-a)
            f.append(axis2EVix(a,z,x,y))
            (x,y) = nextV[(x,y)] 
            f.append(axis2EVix(a,z,x,y))
        Cells[k] = f
    c = []
    for a in range(3):
        for z in range(2):
            f = [axis2EVix(a,z,x,y) for (x,y) in VOrder]
            c.extend(f)
    Cells[(2,2,2)] = c
    return Cells

def Template24CellFacewise():
    '''4-valent cellulation using 24 Vertex Truncated Octahedrons for 3D colour code'''
    VOrder = [(0,0),(0,1),(1,1),(1,0)]
    inner = [(0,2,2)+v for v in VOrder]
    outer = [(2,0,2,1,0),(2,2,0,0,0),(2,1,2,1,0),(2,2,1,0,0)]
    Cells = dict()
    for i in range(4):
        j = (i+1) % 4
        Cells[(0,)+VOrder[i]] = {inner[i],inner[j],outer[i],outer[j]}
    Cells[(0,2,2)] = set(inner+outer)
    return Cells

##########################################################
## Graph Constructions
##########################################################

def SimplexGraph(d):
    '''edge vertex pairs of d-dimensional simplex'''
    A = Mnt(d,2,mink=2)
    return [bin2Set(a) for a in A]

def cycle_graph(d,closed=True):
    '''Cycle graph with d vertices'''
    nRows = d if closed else d-1
    return [(i,(i + 1) % d) for i in range(nRows)]

def star_graph(d):
    n = 2*d + 1
    A = [(i,(i + 1) % n) for i in range(n)]
    B = [(i,(i + d) % n) for i in range(n)]
    return A + B

def oct_graph(d):
    '''graph basd on octahedron'''
    n = 2*d
    # all connected, apart from antipodal points
    return [(i,j) for i in range(n-1) for j in range(i+1,n) if (i + d) % n != j]

def edgeGraph(myrow,d):
    '''given an RG tesselation, return a list of edge vertex pairs, ordered around cycles'''
    CList = RGCellulation(myrow,d)
    E,F = CList[0],CList[1]
    ## set of non-intersecting faces which cover the cell
    FList = FColouring(F)
    Cycles = EF2Cycles(E,FList)
    EList = [e for c in Cycles for e in cycle2edges(c)]
    return EList

def cycle2edges(c):
    '''generate edge vertex pairs from cycle c'''
    cLen = len(c)
    E = []
    for i in range(cLen):
        j = (i + 1) % cLen
        E.append((c[i],c[j]))
    return E

def graphSumm(GList):
    '''Print summary of parameters for a list of graphs'''
    EVec = []
    VVec = []
    for E in GList:
        V = graphVertices(E)
        EVec.append(len(E))
        VVec.append(len(V))
    return f'E={EVec}, V={VVec}'


##########################################################
## Reflection Group Codes
##########################################################
## 2D Tesselations
# myfile = "6-3-codes.txt"
myfile = "8-3-codes.txt"
# myfile = "10-3-codes.txt"
# myfile = "12-3-codes.txt"
# myfile = "14-3-codes.txt"

## 3D Tesselations
## tesselation of 24-cells
# myfile = "3-3-3-3-codes.txt"
# ## tesselation of 48-cells
# myfile = "3-4-3-4-codes.txt"
# # myfile = "4-3-4-2-codes.txt"

## Uncomment to use Reflection Group codes
t = 2 if len(myfile) - len(myfile.replace("-","")) == 2 else 3
codeList = importCodeList(myfile)
codeType = 'RG'

## Uncomment to select a subset of codes
# ix = range(10,15)
# codeList = [codeList[i] for i in ix]

####################################
## Graph Product Codes
####################################

## Hyperbolic Surface Codes:  https://doi.org/10.1088/2058-9565/aa7d3b
## choose to use expander graph
# myfile = "5-4-codes.txt"
# myfile = "6-4-codes.txt"

# expanderList = importCodeList(myfile)

## 2D Codes
# codeType = 'SX'
# t = 2
# TCells = TemplateHex2D()
# # TCells = TemplateSquare2D()
# codeList = []
# for d in range(2,8):
#     # EG = cycle_graph(d)
#     # EG = mgg(d+1)
#     # EG = edgeGraph(expanderList[d-1])
#     EG = star_graph(d)
#     CG = cycle_graph(d)
#     # print('EG',EG)
#     # print('EG',graphSumm([EG]))
#     # SX, SZ = graphProduct2D([cycle_graph(d),EG],TCells)
#     SX = graphProduct2D([CG,CG],TCells)
#     r, n = np.shape(SX)
#     codeList.append(SX)

## 3D Graph Product Codes
# codeType = 'SX'
# t = 3
# TCells = Template24Cell()
# # TCells = Template_12_24_48_Cell()
# codeList = []
# for d in range(2,3):
#     CG = cycle_graph(d,closed=True) 
#     EG = star_graph(d)
#     EG = edgeGraph(expanderList[d-1],2)
#     # EG = oct_graph(3)
#     GList = (CG,CG,EG)
#     print('## ',graphSumm(GList))
#     SX = graphProduct3D(GList,TCells)
#     codeList.append(SX)

## uncomment for code parameters report
# print('n\tk\tdZ\tgamma\tnSX\tnSZ\twLZ\twSX\twSZ')
for myrow in codeList:
    if codeType == 'RG':
        Cells = RGCellulation(myrow, t)
    else:
        Cells = SXCellulation(myrow, t)
    ## uncomment for analysis of LO found by bicolouring
    bicolouringAnalysis(Cells,t)
    ## uncomment for code parameter report
    # print(codeStats(Cells,True,True,False))