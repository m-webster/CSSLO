import itertools as iter
from common import *
from NHow import *
import concurrent.futures

def minWeightZ2(A,tB=1,pList=None):
    '''Find lowest weight generating of <SZ,LZ> which has a non-trivial logical action.
    Uses coset leader method of https://arxiv.org/abs/1211.5568
    Return z component and action'''
    if pList is None:
        pList = pListDefault(tB)
    A = getH(A,2)
    probA = ZMatProb(A,pList,tB,N=2)
    done = False
    while not done:
        done = True
        for i in range(len(A)):
            B = A ^ A[i]
            probB = ZMatProb(B,pList,tB,N=2)
            ix = probB > probA
            ix[i] = False
            if np.any(ix):
                A[ix] = B[ix]
                probA[ix] = probB[ix]
                done = False
    return A

##########################################
## Genetic Algorithm for Distance
##########################################

def distGenetic(L,S=None,tB=1,pList=None,settgs=None):
    k,n = L.shape
    nB = n // tB
    if pList is None:
        pList = pListDefault(tB)
    L = np.hstack([L,ZMatI(k)])
    if S is None:
        S = ZMatZeros((0,n+k))
    else:
        S = np.hstack([S,ZMatZeros((len(S),k))])
    defs = {'lambmu':3,'mu':nB,'pMut':2 / nB,'sMut':1/ (nB * nB),'genCount':1 + int(np.log(nB)),'tabuLength':nB,'fast': True}

    ## used defaults for any settings not in settgs
    if settgs is not None:
        settgs = defs | settgs
    else:
        settgs = defs
    settgs['genCount'] = 1 + int(np.log(nB)) if settgs['fast'] else 1 + int(nB ** 0.5)

    best = None
    population = [np.random.permutation(nB) for i in range(settgs['lambmu']*settgs['mu'])]
    probList,LOList,population = distGeneticEval(L,S,tB,pList,population)
    tabulist = []
    for g in range(settgs['genCount']):
        ix = argsort(probList,reverse=True)[:settgs['mu']]
        population = [distGeneticMutate(population[j],settgs,tabulist) for i in range(settgs['lambmu']) for j in ix]
        probList,LOList,population = distGeneticEval(L,S,tB,pList,population)
        j = np.argmax(probList)
        if best is None or probList[j] > best:
            best = probList[j]
            L = LOList[j]
    L = L[:,:-k]
    return best,L

def indepL(L,pList,nA,tB):
    w = np.sum(L[:,:-nA],axis=-1)
    L = L[w > 0,:]
    probs = ZMatProb(L,pList=pList,nA=nA,tB=tB,N=2)
    ix = argsort(probs)
    probs = probs[ix]
    L = L[ix,:]
    H = L[:,-nA:]
    H,U = getHU(H,2)
    K = U[np.sum(H,axis=-1) == 0,:]
    K,LI = getH(K,2,retPivots=True)
    ix = invRange(len(L),LI)
    return L[ix,:],probs[ix]

def permEval(LS,L,pList,nA,tB,ix):
    L1 = ZMatPermuteCols(LS,ix,nA=nA,tB=tB)
    L1 = getH(L1,N=2,nA=nA,tB=tB)
    ixR = ixRev(ix)
    L1 = ZMatPermuteCols(L1,ixR,nA=nA,tB=tB)
    L1 = np.vstack([L,L1])
    L1,probs1 = indepL(L1,pList,nA,tB)
    return L1,probs1

def distGeneticEval(L,S,tB,pList,population,parallell=False):
    probList = []
    LOList = []
    LS = np.vstack([L,S])
    nA = len(L)
    popList = []
    if parallell:
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            threadFuture = {executor.submit(permEval,LS,L,pList,nA,tB,ix): ix for ix in population}
            for future in concurrent.futures.as_completed(threadFuture):
                L1,probs1 = future.result()
                ix = threadFuture[future]
                LOList.append(L1)
                probList.append(sum(probs1))
                popList.append(ix)
    else:
        for ix in population:
            L1,probs1=permEval(LS,L,pList,nA,tB,ix)
            LOList.append(L1)
            probList.append(sum(probs1))
            popList.append(ix)
    return probList,LOList,popList

def distGeneticMutate(ix,settgs,tabulist):
    n = len(ix)
    done = False
    p = settgs['pMut'] + settgs['sMut'] * np.random.normal()
    while not done:
        ix2 = ZMat(list(range(n)))
        for i in range(int(p*n)+1):
            a,b = np.random.choice(n,size=2,replace=False)
            ix2[a],ix2[b] = ix2[b],ix2[a]
        ix2 = ZMat(ix)[ix2]
        ixtup = tuple(ix2)
        if ixtup not in set(tabulist):
            if len(tabulist) >= settgs['tabuLength']:
                tabulist.pop(0)
            tabulist.append(ixtup)
            done=True
    return ix2

######################################################
## Calculate X and Z distances for CSS codes
######################################################

def minWeight(A,method='gen',tB=1):
    '''Return a set of vectors spanning <SX> which have minimum weight.'''
    A = getH(A,2)
    if method == 'gen':
        return distGenetic(A,tB=tB)
    else:
        return minWeightZ2(A,tB=tB)

def indepLZ(SZ,LZ,tB=1):
    ## get an independent set of LZ
    return lowWeightGens(LZ,SZ,2,tB=tB)

def minWeightLZ(SZ,LZ,tB=1,method='gen'):
    LZ,SZ = HowRes(SZ,LZ,N=2,tB=tB)
    k, n = LZ.shape
    if method == 'gen':
        defs = {'lambmu':3,'mu':n//3,'pMut':2 / n,'sMut':1/ (n * n),'genCount':1 + int(np.log(n)),'tabuLength':10 * n}
        p,LZ = distGenetic(LZ,SZ,settgs=defs)
        return LZ
    else:
        SZLZ = minWeightZ2(np.vstack([SZ,LZ]),tB=tB)
        return indepLZ(SZ,SZLZ,tB=tB)

####################################################
## Zimmerman min distance algorithm
####################################################

def zimDist(C,N=2,tB=1):
    '''Find set of minimum weight generators of linear code defined by gen matrix C'''
    A,kList,ix = zimForm(C,N=N,tB=tB)
    k,n = A.shape
    dU = n
    CC = []
    done = False
    i=0
    while not done:
        ## calculate distance lower bound
        dL = 0
        for kj in kList:
            if k - kj <= i:
                dL += (i + 1 + kj - k)
        ## generate linear combinations of i+1 rows and update distance upper bound
        C = list(iter.combinations(range(k),i+1))
        C = Sets2ZMat(k,C)
        C = matMul(C,A,N)
        W = ZMatWeight(C,tB=tB)
        dU = min(W)
        CC.append(C)
        i += 1
        done = (i == len(kList)) or dU <= dL
    return dU,np.vstack(CC)

def zimForm(A,N=2,tB=1):
    '''Zimmerman form - series of codes with successive information sets of size kj'''
    kList = []
    m,n = A.shape
    nA,nB,nC = blockDims(n,tB=tB)
    ix = ZMat2D(np.arange(nB))
    if np.sum(A) == 0:
        return A,kList,ix
    A = A.copy()
    k = nB
    done = False
    while not done:
        A,LI = getH(A,N,tB=tB,nC=k,retPivots=True)
        # LI = HowPivots(A,tB=tB,nC=k)
        LI = inRange(k,LI)
        ixk = np.hstack([invRange(k,LI), LI])
        ix = ZMatPermuteCols(ix,ixk,tB=1)
        A = ZMatPermuteCols(A,ixk,tB=tB)
        k = k - len(LI)
        kList.append(len(LI))
        w = ZMatWeight(A,tB=tB,nC=k)
        done = (np.sum(w) == 0)
    kList.reverse()
    ix = ix[0]
    ixR = ixRev(ix)
    A = ZMatPermuteCols(A,ixR,tB=tB)
    return A,kList,ix

def GRANDdX(SZ,LZ,maxd=None):
    '''X-distance using GRAND method'''
    r,n = np.shape(LZ)
    maxd = n if maxd is None else maxd
    for d in range(maxd):
        ## all binary vectors of weight d
        for x in iter.combinations(range(n),d):
            x = set2Bin(n,x)
            ## syndrome vector - if a stabiliser or LX, this should be zero
            s = np.mod(SZ @ x, 2)
            if np.sum(s) == 0:
                ## if LX, this should be non-zero
                s = np.mod(LZ @ x, 2)
                if np.sum(s) > 0:
                    return d
    return maxd