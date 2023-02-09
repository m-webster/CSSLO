import numpy as np
from NSpace import *
from common import * 

########################################
## Algebra of CP operators            ##
########################################

def XCPMul(A, B, V, N,CP=True):
    '''multiply CP operators A, B uses CPX'''
    x1, z1 = A 
    x2, z2 = B 
    c = CPX(x2,z1,V,N,CP)
    x = np.mod(x1 + x2, 2)
    z = np.mod(c + z2, 2 * N)
    return (x, z)


def CPComm(z,x,V,N,CP=True):
    '''Commutator of CP_N,V(z) and x'''
    ## [[X,CP(z)]] = X CP(z) X CP(-z) = X X CPX(x,z) CP(-z) = CPX(x,z) CP(-z)
    c = CPX(x,z,V,N,CP)
    return np.mod(c - z, 2*N)


def CPX(x,z,V,N,CP=True):
    '''Fundamental commutation relation for CP operators
    calculate diagonal component of CP(z) X'''
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


def tupleDict(V):
    '''Convert V to a dictionary indexed by tuple(v)'''
    m,n = np.shape(V)
    return {tuple(V[i]): i for i in range(m)}


def RPX(x,q,V,N):
    '''Fundamental commutation relation for RP operators
    calculate diagonal component of RP(q,V) X'''
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


def pIndex(V):
    '''Find phase component - corresponds to all zero vector, usually in last position'''
    j = np.where(np.sum(V,axis=-1) == 0)
    return j[0] if len(j) == 1 else None


def CPSetV(q1,V1,V2):
    '''Take CP(q1,V1) and convert to CP(q2,V2)'''
    vDict = tupleDict(V2)
    q2 = ZMatZeros(len(V2))
    for q, v in zip(q1,V1):
        v = tuple(v)
        if v not in vDict:
            ## error!
            return None
        q2[vDict[v]] = q 
    return q2


def CPNonZero(q1,V1):
    '''Eliminate rows v from V1 and cols from q1 where q1[v] is zero'''
    q1 = ZMat(q1)
    V1 = ZMat(V1)
    ix = [i for i in range(len(q1)) if q1[i] != 0]
    return q1[ix],V1[ix]



########################################
## String representation of operators ##
########################################


def x2Str(x):
    '''String rep of X component'''
    if np.sum(x) == 0:
        return ""
    xStr = [f'[{i}]' for i in bin2Set(x)]
    return f' X{"".join(xStr)}'


def z2Str(z,N,phase=False):
    '''Represent Z component as string'''
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



def CP2Str(qVec,V,N,CP=True):
    '''String representation of CP operator'''
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


def XCP2Str(a,V,N):
    '''String rep of operator with X and CP components'''
    x,qVec = a
    pStr, zStr = CP2Str(qVec,V,N)
    return pStr + x2Str(x) + zStr


def Str2CP(mystr,n=None,t=None,CP=True):
    '''Convert string to XCP operator'''
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
    minn = (max(minn) if len(minn) > 0 else 0) + 1
    mint = max(mint) if len(mint) > 0 else 1
    n = minn if n is None else n 
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
        if f >= 0:
            j = VDict[tuple(set2Bin(n,v))]
            qVec[j] += (num * (1 << f))
    qVec = np.mod(qVec,2*N)
    return (x, qVec), V, t


###################################################
## Duality between CP and RP
###################################################


def uLEV(u,V):
    '''Return binary vector b[v] = 1 iff u <= v'''
    ix = bin2Set(u)
    return np.prod(V.T[ix],axis=0)


def uGEV(u,V):
    '''Return binary vector b[v] = 1 iff u >= v'''
    m,n = np.shape(V)
    ## weights of intersections of u with V
    W = (ZMat([u]) @ V.T)[0]
    ## ix indicates where the W matches weight of V
    ix = (W == np.sum(V,axis=-1))
    ## all zero vector, apart from where weights match
    temp = ZMatZeros(m)
    temp[ix] = 1
    return temp   


def getV2(q1,V1,t):
    '''V2 is the target for application of CP2RP'''
    m,n = np.shape(V1)
    temp = [bin2Set(v) for v in V1]
    return Mnt_partition(temp,n,t)


def CP2RP(q1,V1,t,CP=True,Vto=None):
    '''Convert RP(q1,V1) to CP(q2,V2)) and vice versa'''
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


def Mnt(n,t,mink=1):
    '''Rows are binary strings of length n of weight mink to t'''
    A = [set2Bin(n,s) for k in range(mink, t+1) for s in itertools.combinations(range(n),k)]
    return ZMat(A)


def MntSubsets(S,t,mink=1):
    '''Return list of subsets of max weight t whose support is a subset of S'''
    return {s for k in range(mink, t+1) for s in itertools.combinations(S,k)}


def Mnt_partition(cList,n,t,mink=1):
    '''Same as Mnt - but this time using a partition
    cList is a list of partitions
    max weight of vector in Mrm is t'''
    temp = set()
    for c in cList:
        temp.update(MntSubsets(c,t,mink))
    temp = [(n-len(s),tuple(set2Bin(n,s))) for s in temp]
    temp = sorted(temp,reverse=True)
    return ZMat([s[1] for s in temp])


def CP2Partition(qList,V):
    '''Partition from non-zero elements of qList'''
    m,n = np.shape(V)
    qList, V = CPNonZero(qList, V)
    addSupp = bin2Set(1 - np.sum(V,axis=0))
    return [bin2Set(v) for q,v in zip(qList,V)] + [[i] for i in addSupp]


def CPlevel(q,V,N,CP=True):
    '''Calculate Clifford level of CP or RP operator'''
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
    