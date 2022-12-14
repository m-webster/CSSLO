import numpy as np
import sys
import itertools
import time

## object helper functions
## simplify display of complex np.arrays
# np.set_printoptions(precision=3,suppress=True)

# def argsort(seq):
#     # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
#     return sorted(range(len(seq)), key=seq.__getitem__)

from collections.abc import Mapping, Container
from sys import getsizeof
 
def deep_getsizeof(o, ids=None):
    if ids is None:
        ids = set()
    d = deep_getsizeof
    if id(o) in ids:
        return 0
 
    r = getsizeof(o)
    ids.add(id(o))
 
    if isinstance(o, str) or isinstance(0, str):
        return r
    
    if isinstance(o, dict):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())
 
    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
 
    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)
 
    return r 

def currTime():
    return time.process_time()

def startTimer():
    global startTime
    startTime = currTime()
    return startTime

def elapsedTime():
    global startTime
    return -startTime + startTimer()

def func_name():
    """
    :return: name of caller
    """
    return sys._getframe(1).f_code.co_name

def typeName(val):
    return type(val).__name__


#######################################
####            ZMat               ####
#######################################
## Integer Matrices

## make an integer numpy array
## if n is set, check that rows have length n
def ZMat(A,n=None):
    if typeName(A) in ['set','range']:
        A = list(A)
    if typeName(A) != 'ndarray' or A.dtype != int:
        A = np.array(A,dtype=int)
    if n is not None:
        s = list(A.shape)
        if s[-1] == 0:
            A= np.empty((0,n),dtype=int)
    return A

## ensure A is a 2-dimensional integer numpy array
def ZMat2D(A):
    A = ZMat(A)
    if np.ndim(A) == 2:
        return A
    if np.ndim(A) == 0:
        return ZMat([[A]])
    if np.ndim(A) == 1:
        return ZMat([A])    
    d = np.shape(A)[-1]
    return np.reshape(A,(-1,d))

## ensure A is a 1-dimensional integer numpy array of length n
def ZMat1D(A,n):
    A = ZMat(A)
    if np.ndim(A) != 1:
        A = np.reshape(A,(-1,))
    if len(A) < n:
        A = np.tile(A,n//len(A))
    if len(A) > n:
        A = A[:n]
    return A

def row2components(r):
    ix = ZMat(np.nonzero(r))
    return ix, r[ix]

def ZMat2List(r):
    r = np.reshape(r,(-1))
    return list(r)

def row2compStr(r):
    ix,vals = row2components(r)
    # ix = ";".join(ZMat2str(ix))
    # vals = ";".join(ZMat2str(vals))
    ix = str(ZMat2List(ix))
    vals = str(ZMat2List(vals))
    return f'{ix}:= {vals}'.replace(" ","")

def ZMat2compStr(A):
    return "\n".join([row2compStr(r) for r in A])

# Input: string of single digit numbers, split by spaces
# Output: integer array
def str2ZMat(mystr):
    if mystr.find(" ") > 0:
        mystr = mystr.split()
    # print(func_name(),mystr)
    return ZMat([int(s) for s in mystr])


def str2ZMatdelim(S=''):
    sep=','
    for s in "\r\n;":
        S = S.replace(s,sep)
    S = S.split(sep)
    # print(func_name(),S)
    return ZMat([str2ZMat(s) for s in S])

def int2ZMat(A,N=2,n=None):
    ## return an array representation of integer x
    ## x has n bits in base N
    ## need to cover situation where array is empty
    if n is None:
        n = logCeil(np.amax(A),N)
    if np.size(A) == 0:
        return ZMat(emptyadj(A,1),n)
    d = np.ndim(A)
    B = np.expand_dims(A, axis=d)
    B = np.repeat(B,n,axis=d)
    Ni = N ** np.arange(n-1,-1,-1)
    return np.apply_along_axis(func1d=modDiv,axis=d,arr=B,b=Ni,N=N)


## handle multiple types of binary vector input
## return ZMat
def bin2Zmat(SX):
    if SX is None:
        return SX 
    ## convert string to ZMat
    if isinstance(SX,str):
        return str2ZMatdelim(SX)
    ## convert array to ZMat
    return ZMat(SX)

## vector length n - SX could be None
def binn(SX):
    if SX is None:
        return 0 
    SX = ZMat2D(SX) 
    m,n = np.shape(SX)
    return n

def modDiv(a,b,N):
    return np.mod(a//b,N)

def emptyadj(A,n):
    if n ==0:
        return A
    s = list(np.shape(A))
    if n+len(s) < 0:
        return ZMat([])
    if n > 0:
        s =  s + [0]*n
    if n < 0:
        s = s[:n + len(s) +1]
    return np.reshape(A,s)

def ZMat2int(A,N=2):
    A = ZMat(A)
    ## need to cover situation where array is empty
    if np.size(A) == 0:
        return emptyadj(A,-1)    
    n = np.shape(A)[-1]
    Ni = N ** np.arange(n-1,-1,-1)
    return np.apply_along_axis(func1d=np.dot,axis=-1,arr=A,b=Ni)

## print table, inserting dividers at rowdiv, coldiv
def ZmatTable(data,rowdiv=None,coldiv=None):
    # get max lengths of columns
    colLen = np.amax(np.char.str_len(data),axis=0)
    m,n = np.shape(data)
    for i in range(n):
        data[:,i] = np.char.rjust(data[:,i],colLen[i])
    # insert column dividers
    if coldiv is not None:
        mycol = "|"
        data = np.insert(data,coldiv,mycol,axis=-1)
    # merge rows with " " separators
    data = np.array([" ".join(data[i]) for i in range(m)])
    # insert row dividers
    if rowdiv is not None:
        myrow = "-" * len(data[0])
        data = np.insert(data,rowdiv,myrow,axis=0)
    # join rows 
    return "\n".join(data)

def ZMat2str(A,N=None):
    if np.size(A) == 0:
        return ""
    S = np.char.mod('%d', A)
    sep = ""
    if N is None:
        N = np.amax(A) + 1
    if N > 10:
        Nw= len(str(N-1))
        S = np.char.rjust(S,Nw)
        sep = " "
    return np.apply_along_axis(func1d=sepjoin,axis=-1,arr=S,sep=sep)

def sepjoin(a,sep):
    return sep.join(a)

def ZmatPrint(A,N=None):
    return "\n".join(ZMat2str(ZMat2D(A),N))

def isiter(vals):
    # can we turn vals into an array?
    return hasattr(vals,'count') or hasattr(vals,'__array__')

def isint(val):
    v = typeName(val)
    return v[:3] == "int"
    return val == int(val)



## is x a power of N?
def isPower(x,N=2):
    t = logCeil(x,N)
    return N ** (t-1) == x

## return max(t) where x = N^t
def logCeil(x,N=2):
    i = 0
    while x > 0:
        x = x // N
        i = i+1
    return i

## find t such that N = 2**t or None otherwise
def log2int(N):
    t = logCeil(N-1,2) 
    return t if 2 ** t == N else None

## argsort but allowing for sorting of tuples
def argsort(seq,reverse=False):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__,reverse=reverse)

def argmin(seq,reverse=False):
    return max(range(len(seq)), key=seq.__getitem__) if reverse else min(range(len(seq)), key=seq.__getitem__) 

## optimize
def weight(A):
    return sum([1 if a > 0 else 0 for a in A])

def getmn(A):
    n = len(A)
    m = len(A[0]) if n else 0
    return m,n

def makeList(A):
    return rowVector(A)[0]

def ZMatAddZeroRow(A):
    ## append a row of all zeros at the end of A
    A = ZMat2D(A)
    d = np.shape(A)[-1]
    return np.vstack([A,ZMatZeros(d)])

def RemoveZeroRows(A,N=False):
    ## remove any zero rows
    A = ZMat2D(A)
    ix = np.logical_not([isZero(a,N) for a in A])
    return A[ix]

def RemoveZeroCols(A,N=False):
    A = ZMat2D(A)
    m,n = np.shape(A)
    ix = np.logical_not([isZero(A[:,i],N) for i in range(n)])
    return A[:,ix]

def isConstant(A):
    return np.amax(A) == np.amin(A)

def isZero(A,N=False):
    ## check if the row is all zeros % N
    if N:
        A = np.mod(A,N)
    return np.all(A == 0)

def ZMatI(n):
    ## identity n x n matrix
    return np.eye(n,dtype=int)

## return np array of zeros of length/shape s
def ZMatZeros(s):
    return np.zeros(s,dtype=int)

## sort list of XP operators 
def ZMatSort(A,reverse=False):
    if np.ndim(A) == 1:
        return A
    A = ZMat(A)
    ix = argsort(ZMat2tuple(A),reverse=reverse)
    return A[ix,:]

def ZMat2tuple(A):
    # print(func_name(),A)
    n = np.shape(A)[-1]
    A = np.reshape(A,(-1,n))
    return [tuple(a) for a in A]

## check if two sets of XP operators A and B are equal
def ZMatEqual(A,B):
    if np.shape(A) != np.shape(B):
        return False
    A,B = ZMatSort(A),ZMatSort(B)

    return np.array_equal(A,B)

def leadingIndex(a):
    # report('a',a)
    i = 0
    n = len(a)
    while i < n and a[i]==0:
        i+=1
    return i

def set2Bin(n,t):
    temp = ZMatZeros(n)
    temp[list(t)] = 1
    return temp

def bin2Set(v):
    return [i for i in range(len(v)) if v[i] != 0]


def leadingIndices(K):
    n = len(K)
    if n == 0:
        return []
    L = [len(K[0])] * (n)
    m = len(K[0])
    i = 0
    j = 0
    while i < m and j < n:
        if K[j][i] == 0:
            i += 1
        else:
            L[j] = i
            j+=1
            # i+=1
    return L

## for binary matrices of form IA return kernel 
## Not sure if it works perfectly for N>2
def kerIA(M,N=None,C=None):
    # print(func_name(),'M')
    # print(ZmatPrint(M))
    if N is None:
        N =2
    if C is not None:
        Z = matMul(C,np.transpose(M),N)
        return np.all(Z==0)
    r,n = np.shape(M)
    if (r == n):
        return ZMatZeros((0,n))
    L = leadingIndices(M)
    nL = [i for i in range(n) if i not in set(L)]
    MnL = (N-1)*M[:,nL]
    Inr = np.diag([1]*(n-r))
    temp = [[] for i in range(n)]
    for i in range(r):
        temp[L[i]] = MnL[i]
    for i in range(n-r):
        temp[nL[i]] = Inr[i]
    return np.transpose(temp)

    r,n = np.shape(M)
    temp = []
    for r in M:
        s = bin2Set(r)
        l = s.pop()
        for j in s:
            rj = set2Bin(n,[j])
            rj[l] = N-1
            temp.append(rj)
    ## columns where there are no entries
    s = np.sum(M,axis=0)
    s = [i for i in range(len(s)) if s[i]==0]
    for j in s:
        temp.append(set2Bin(n,[j]))
    return ZMat(temp)



    

    # At = np.transpose(M[:,r:])
    # if N is not None:
    #     At = (N-1)*At
    # return np.hstack([At ,np.eye(n-r,dtype=int)]) 

## iterator - rows correspond to subsets of [0..n-1] of size between w1 and w2
def BinPowerset(n,w1=None,w2=None):
    w1 = n if w1 is None else min(w1,n)
    wrange = range(w1+1) if w2 is None else range(w1,w2+1)
    for w in wrange:
        for t in itertools.combinations(range(n),w):
            yield set2Bin(n,t) 


def RowSpan(A,N=2):
    A = ZMat(A)
    g = ZMat([np.lcm.reduce(N // np.gcd(a,N)) for a in A])
    # report('g',g)
    G = [range(a) for a in g]
    for ix in itertools.product(*G):
        # ix = list(ix)
        yield np.mod(ix @ A,N)

# def SS2row(n,a):
#     b = np.zeros(n,dtype=int)
#     b[list(a)] = 1
#     return b

# def RowSpanComb(A,N=2,wMin=0,wMax=None,returnIx=False):
#     A = ZMat(A)
#     n = len(A)
#     if wMax is None:
#         wMax = n
#     for m in range(wMin,wMax+1):
#         for a in itertools.combinations(range(n),m):
#             ix = SS2row(n,a)
#             x= np.mod(ix @ A,N)
#             if returnIx:
#                 yield x,ix
#             else:
#                 yield x

def colVector(b):
    return  np.reshape(ZMat2D(b),(-1,1))

def rowVector(b):
    return  np.reshape(ZMat2D(b),(1,-1))

def matEqual(A,B,N):
    temp = matAdd(A,-B,N)
    return isZero(temp,N)

def matAdd(A,B,N):
## resize to allow valid addition
    ma,na = A.shape
    mb,nb = B.shape
    m = max(ma,mb)
    n = max(na,nb)
    A = matResize(A,m,n)
    B = matResize(B,m,n)
    return np.mod(A+B,N)

def matMul(A,B,N=False):
    A = ZMat2D(A)
    B = ZMat2D(B)
    ma,na = A.shape
    mb,nb = B.shape
    n = max(na,mb)
    if na < n:
        A = matResize(A,ma,n)
    if mb < n:
        B = matResize(B,n,nb)
    if N is False:
        return A @ B
    else:
        return np.mod(A @ B, N)
    
# def ZMat2D(A,square=True):
#     ## convert general object into np.array over int ##
#     A = np.array(A,dtype=int)
#     if len(A) == 0:
#         return np.array([[]])
#     s = A.shape
#     if square and len(s) == 1:
#         A = np.array([A],dtype=int)
#     return A

def matResize(A,m,n,check=False):
    if check is not False:
        ma,na = check.shape
        return ma == m and na == n
    ma,na = A.shape
    dn = n-na
    dm = m-ma
    if dn == 0 and dm == 0:
        return A
    temp = []
    for i in range(min(m,ma)):
        r = mat2list(A[i])
        if dn > 0:
            r = r + [0] * dn
        if dn < 0:
            r = r[:n]
        temp.append(r)
    for i in range(dm):
        temp.append([0] * n)
    return ZMat2D(temp)

def mat2list(b):
    b = ZMat2D(b)
    return b.flatten().tolist()

# def arr2str(r):
#     return "".join([str(a) for a in r])

# def ZMat2str(A,sep='\n'):
#     temp = [arr2str(r)  for r in A]
#     return sep.join(temp)

## optimise
def Highbit(r):
# print('r',r)
    i = 0
    while i < len(r) and r[i] == 0:
        i += 1
    return i if i < len(r) else -1

# def matDiag(A, N):
#     m,n = A.shape
#     if not m:
#         return A
#     A = np.mod(A,N)
#     temp = rowVector([0])
#     temp = matResize(temp,n)
#     for i in range(len(A)):
#         r = A[i]
#         ix = Highbit(r)
#         if ix > -1:
#             temp[ix] = r
#     return temp

# def indepSet(A,N):
#     temp = []
#     for b in A:
#         r,a = matResidual(A,b,N)
#         if not isZero(r,N):
#             temp.append(r)
#     return temp

###### Debugging #############

verbose = False

def report(*args ):
    ## print, but only if global verbose setting is True
    global verbose
    if verbose:
        print(*args)

verbose_old = []

def setVerbose(val):
    ## set verbose variable, keep previous value
    global verbose, verbose_old
    verbose_old.append(verbose)
    verbose = val

def unsetVerbose():
    ## return verbose to previous setting
    global verbose, verbose_old
    if len(verbose_old):
        verbose = verbose_old.pop() 

def getVerbose():
    global verbose
    return verbose

### variable storage ####

def getVal(obj,label):
    # report('getVal',label)
    
    if checkVal(obj,label):
        return getattr(obj,label)
    getlabel = 'get'+label
    if getlabel not in dir(obj):
        return False
    f = obj.__getattribute__(getlabel)
    if type(f).__name__ != 'method':
        return False
    return setVal(obj,label,f())

def checkVal(obj,label):
    if not hasattr(obj,label):
        return False
    return getattr(obj,label) is not None

def setVal(obj,label,val=None):
    setattr(obj,label,val)
    return val

def setVals(obj,labels,vals):
    return [setVal(obj,labels[i],vals[i]) for i in range(len(labels))]

def getVals(obj,labels):
    return [getVal(obj,label) for label in labels]

# N = 8
# n = 3
# A = np.random.randint(N,size=(n,n))
# B = int2ZMat(A)
# print(A)
# print(B)
# print(ZMat2int(B,2))
# print(ZMat2str(A))

# print(ZMat2D(B))

# D = ZMat([],n)
# D = ZMatAddZeroRow(D)
# print(D)
# print(ZMatRemoveZeroRows(D))

# A = ZMatAddZeroRow(A)
# print(A)
# A = ZMatRemoveZeroRows(A)
# print(A)