import numpy as np
from NSpace import *
from common import * 

########################################
## Algebra of CP operators            ##
########################################

## multiply CP operators A, B
## uses CPX
def XCPMul(A, B, V, N,CP=True):
    x1, z1 = A 
    x2, z2 = B 
    c = CPX(x2,z1,V,N,CP)
    x = np.mod(x1 + x2, 2)
    z = np.mod(c + z2, 2 * N)
    return (x, z)

## commutator of CP_N,V(z) and x
def CPComm(z,x,V,N,CP=True):
    ## [[X,CP(z)]] = X CP(z) X CP(-z) = X X CPX(x,z) CP(-z) = CPX(x,z) CP(-z)
    c = CPX(x,z,V,N,CP)
    return np.mod(c - z, 2*N)

## Fundamental commutation relation for CP operators
## calculate diagonal component of CP(z) X
def CPX(x,z,V,N,CP=True):
    if not CP:
        return RPX(x,z,V,N)
    z = ZMat(z.copy())
    VDict = tupleDict(V)
    for i in bin2Set(x):
        ix = V.T[i] * z
        for j in bin2Set(ix):
            vi = V[j].copy()
            vi[i] = 0
            k = VDict[tuple(vi)]
            z[k] = np.mod(z[k] + z[j],2*N)
            z[j] = np.mod(-z[j],2*N) 
    return z

## convert V to a dictionary indexed by tuple(v)
def tupleDict(V):
    m,n = np.shape(V)
    return {tuple(V[i]): i for i in range(m)}

## Fundamental commutation relation for RP operators
## calculate diagonal component of RP(q,V) X
def RPX(x,q,V,N):
    q1 = q 
    p1 = 0
    ## calculate q-vec and phase adj for each X_i
    for i in bin2Set(x):
        qv = V.T[i] * q1
        ## flip sign of q1 
        q1 = np.mod(q1 - 2 * qv,2*N)
        ## accumulate phase
        p1 = p1 + np.sum(qv)
    ## update phase
    j = pIndex(V)
    if j is not None:
        q1[j] = np.mod(q1[j] + p1, 2*N)
    return q1

## find phase component - usually in last position
def pIndex(V):
    j = np.where(np.sum(V,axis=-1) == 0)
    return j[0] if len(j) == 1 else None

## take CP(q1,V1) and convert to CP(q2,V2)
def CPSetV(q1,V1,V2):
    vDict = tupleDict(V2)
    q2 = ZMatZeros(len(V2))
    for q, v in zip(q1,V1):
        v = tuple(v)
        if v not in vDict:
            ## error!
            return None
        q2[vDict[v]] = q 
    return q2

## Eliminate rows v from V1 and cols from q1 where q1[v] is zero
def CPNonZero(q1,V1):
    q1 = ZMat(q1)
    V1 = ZMat(V1)
    ix = [i for i in range(len(q1)) if q1[i] != 0]
    return q1[ix],V1[ix]



########################################
## String representation of operators ##
########################################

## string rep of X component
def x2Str(x):
    if np.sum(x) == 0:
        return ""
    xStr = [f'[{i}]' for i in bin2Set(x)]
    return f' X{"".join(xStr)}'

## rep Z component as string
def z2Str(z,N,phase=False):
    syms = {1:"I",2:'Z',4:'S',8:'T',16:'U'}
    g = np.gcd(z,  N)
    den = (N)//g 
    num = z//g 
    temp = [(den[i],num[i],i) for i in range(len(z)) if num[i] > 0]
    zStr = ""
    pStr = ""
    lastsym = ""
    for d,n,i in sorted(temp):
        if phase and i == len(z) - 1:
            pStr = f'w{n}/{d}' if d > 1 else ""
        else: 
            if n == 1:
                n = ""
            nextsym = f' {syms[d]}{n}'
            if nextsym != lastsym:
                lastsym = nextsym
                zStr += nextsym
            zStr += f'[{i}]'   
    if len(zStr) == 0:
        zStr = " I"    
    return pStr + zStr


## String rep of CP operator
def CP2Str(qVec,V,N,CP=True):
    qVec, V = CPNonZero(qVec,V)
    opSym = 'C' if CP else 'R'
    syms = {1:"I",2:'Z',4:'S',8:'T',16:'U'}
    g = np.gcd(qVec, 2 * N)
    den = (2 * N)//g 
    num = qVec//g 
    m,n = np.shape(V)
    w = np.sum(V,axis=-1)
    temp = [(n-w[i],den[i],num[i],tuple(V[i])) for i in range(len(qVec)) if num[i] > 0]
    temp = sorted(temp)
    i = len(temp)
    zStr = ""
    pStr = ""
    lastsym = ""
    while i > 0:
        i -= 1
        w,den,num,v = temp[i]
        w = n-w
        # print(" w,d,n,v ", w,d,n,v )
        if w == 0:
            pStr = f'w{num}/{den}' if den > 1 else ""
        else:
            if num == 1:
                num = ""
            nextsym = f' {opSym*(w-1)}{syms[den]}{num}'
            if nextsym != lastsym:
                lastsym = nextsym
                zStr += nextsym
            zStr += str(bin2Set(v)).replace(" ","")  
    if len(zStr) == 0:
        zStr = " I"    
    return pStr, zStr

## string rep of operator with X and CP components
def XCP2Str(a,V,N):
    x,qVec = a
    pStr, zStr = CP2Str(qVec,V,N)
    return pStr + x2Str(x) + zStr

## convert string to XCP operator
def Str2CP(mystr,n=None,t=None,CP=True):
    mystr = mystr.upper()
    repDict = {' ':'','C':'','R':'',']':'*','[':'*',',':' '}
    for a,b in repDict.items():
        mystr = mystr.replace(a,b)
    # print('mystr',mystr)
    mystr = mystr.split('*')
    # print('mystr',mystr)
    xList,CPList,minn,mint = [],[],[],[]
    symDict = {'X':1,'Z':1, 'S':2,'T':3,'U':4,'W':0}
    for a in mystr:
        if (len(a)) > 0:
            s = a[0]
            if s in symDict:
                den = symDict[s]
                num = 1 if len(a) == 1 else int(a[1:])
                xcomp = s == 'X'
                if s == 'W':
                    j = a.find('/')
                    if j > 0:
                        num = int(a[1:j])
                        den = int(a[j+1:])
                        CPList.append((num,den,[]))
            else:
                # v = str2ZMat(a)
                v = [int(b) for b in a.split(" ")]
                minn.extend(v)
                if xcomp:
                    xList.append(v[0])
                else:
                    ti = den + len(v) - 1 if CP else den
                    mint.append(ti)
                    CPList.append((num,den,v))
    # print('x',xList)
    # print('z',CPList)
    minn = (max(minn) if len(minn) > 0 else 0) + 1
    mint = max(mint) if len(mint) > 0 else 1
    # print('minn',minn)
    # print('mint',mint)
    n = minn if n is None else n 
    # f = 1 if n is None else 1 << (t - mint )
    t = mint if t is None else t
    N = 1 << t
    V = Mnt(n,t,mink=0)
    m = len(V)
    VDict = tupleDict(V)
    qVec = ZMatZeros(m)
    x = ZMatZeros(n)
    for i in xList:
        x[i] += 1
    x = np.mod(x,2)
    for (num,den,v) in CPList:
        f = t + 1 - den - len(v)
        f = t + 1 - den
        # print('f,t,den,len(v)',f,t,den,len(v))
        if f >= 0:
            j = VDict[tuple(set2Bin(n,v))]
            qVec[j] += (num * (1 << f))
    qVec = np.mod(qVec,2*N)

    # print('x',x)
    # print('z',z)
    # print('V')
    # print(ZmatPrint(V))
    return (x, qVec), V, t


###################################################
## Duality between CP and RP
###################################################

## return binary vector b[v] = 1 iff u <= v
def uLEV(u,V):
    ix = bin2Set(u)
    return np.prod(V.T[ix],axis=0)

## return binary vector b[v] = 1 iff u >= v
def uGEV(u,V):
    m,n = np.shape(V)
    ## weights of intersections of u with V
    W = (ZMat([u]) @ V.T)[0]
    ## ix indicates where the W matches weight of V
    ix = (W == np.sum(V,axis=-1))
    ## all zero vector, apart from where weights match
    temp = ZMatZeros(m)
    temp[ix] = 1
    return temp   

## V2 is the target for application of CP2RP
def getV2(q1,V1,t):
    m,n = np.shape(V1)
    temp = [bin2Set(v) for v in V1]
    return Mnt_partition(temp,n,t)

## convert RP(q1,V1) to CP(q2,V2)) and vice versa
def CP2RP(q1,V1,t,CP=True,Vto=None):
    # print(func_name())
    V2 = getV2(q1,V1,t) if Vto is None else Vto 
    m2, n = np.shape(V2)
    q2 = ZMatZeros(m2)
    w2 = np.sum(V2,axis=-1)
    w1 = np.sum(V1,axis=-1) if CP else w2
    CP_scale = 1 << w1
    alt = (-1) ** (w2 + 1)
    for i in bin2Set(q1):
        ## scaling factor: divide for CP -> RP; multiply for RP -> CP
        w_scale = q1[i] * 2 // CP_scale[i] if CP else q1[i] * CP_scale // 2
        q2 +=  w_scale * uGEV(V1[i],V2) * alt
    q2 = np.mod(q2,1 << (t+1))
    ## set the phase component to be the same as z
    j = pIndex(V1)
    if j is not None:
        q2[j] = q1[j]
    if Vto is None:
        # print(func_name(),'len(V2) before',len(V2))
        q2,V2 = CPNonZero(q2,V2)
    # print('len(V2)',len(V2))
    return (q2,V2)


##########################################
## Embedding Operations
##########################################

## columns are binary reps of subsets of n of size 1 to t
def Mnt(n,t,mink=1):
    A = [set2Bin(n,s) for k in range(mink, t+1) for s in itertools.combinations(range(n),k)]
    return ZMat(A)

## list of subsets of max weight t whose support is a subset of S
def MntSubsets(S,t,mink=1):
    return {s for k in range(mink, t+1) for s in itertools.combinations(S,k)}

## Mnt - but this time using a partition
## cList is a list of partitions
## max weight of vector in Mrm is t
def Mnt_partition(cList,n,t,mink=1):
    temp = set()
    for c in cList:
        temp.update(MntSubsets(c,t,mink))
    temp = [(n-len(s),tuple(set2Bin(n,s))) for s in temp]
    temp = sorted(temp,reverse=True)
    # return np.flip([s[1] for s in temp],axis=0)
    return ZMat([s[1] for s in temp])

## partition from non-zero elements of qList
def CP2Partition(qList,V):
    m,n = np.shape(V)
    qList, V = CPNonZero(qList, V)
    addSupp = bin2Set(1 - np.sum(V,axis=0))
    return [bin2Set(v) for q,v in zip(qList,V)] + [[i] for i in addSupp]

## Calculate Clifford level of CP or RP operator
def CPlevel(q,V,N,CP=True):
    q,V = CPNonZero(q,V)
    t = log2int(N)
    if t is None:
        return t
    den = (2 * N) // np.gcd(q, 2 * N)
    if CP:
        mint = [log2int(den[i]) + np.sum(V[i]) - 1 for i in range(len(q))]
    else:
        mint = [log2int(den[i]) for i in range(len(q))]
    return np.max(mint)
    