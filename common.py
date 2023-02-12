import numpy as np
import sys
import itertools
import time


#######################################
## ZMat - Integer Matrices
#######################################

#######################################
## Create ZMat
#######################################

def ZMat(A,n=None):
    '''Create an integer numpy array. If n is set, ensure that the row length is n.'''
    if typeName(A) in ['set','range']:
        A = list(A)
    if typeName(A) != 'ndarray' or A.dtype != int:
        A = np.array(A,dtype=int)
    if n is not None:
        s = list(A.shape)
        if s[-1] == 0:
            A= np.empty((0,n),dtype=int)
    return A

def ZMatI(n):
    '''Identity n x n integer matrix'''
    return np.eye(n,dtype=int)

def ZMatZeros(s):
    '''Return integer array of zeros of length/shape s'''
    return np.zeros(s,dtype=int)

def set2Bin(n,t):
    '''Convert list of integers t to a binary vector of length n'''
    temp = ZMatZeros(n)
    temp[list(t)] = 1
    return temp

def bin2Set(v):
    '''Convert binary vector to a list of indices such that v[i] !=0'''
    return [i for i in range(len(v)) if v[i] != 0]

def colVector(b):
    return  np.reshape(ZMat2D(b),(-1,1))

def rowVector(b):
    return  np.reshape(ZMat2D(b),(1,-1))

####################################
## Operations on ZMat
####################################

def matAdd(A,B,N):
    '''Add two integer matrices, first resizing to allow valid addition'''
    ma,na = A.shape
    mb,nb = B.shape
    m = max(ma,mb)
    n = max(na,nb)
    A = matResize(A,m,n)
    B = matResize(B,m,n)
    return np.mod(A+B,N)

def matMul(A,B,N=False):
    '''Multiply two integer matrices modulo N. Reshape to allow valid multiplication.'''
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

def RemoveZeroRows(A,N=False):
    '''Remove any zero rows from integer matrix A'''
    A = ZMat2D(A)
    ix = np.logical_not([isZero(a,N) for a in A])
    return A[ix]

def ZMatSort(A,reverse=False):
    '''Sort rows of A, considering each row to be a tuple.'''
    if np.ndim(A) == 1:
        return A
    A = ZMat(A)
    ix = argsort(ZMat2tuple(A),reverse=reverse)
    return A[ix,:]

def ZMat2tuple(A):
    '''Convert rows of A to tuples.'''
    n = np.shape(A)[-1]
    A = np.reshape(A,(-1,n))
    return [tuple(a) for a in A]

def argsort(seq,reverse=False):
    '''Argsort but allowing for sorting of tuples'''
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__,reverse=reverse)

##############################
## ZMat Info
##############################

def matEqual(A,B,N):
    '''Check if integer matrices A and B are equal.'''
    temp = matAdd(A,-B,N)
    return isZero(temp,N)

def leadingIndex(a):
    '''Return leading index of vector a (ie smallest value for which a[i] !=0)'''
    i = 0
    n = len(a)
    while i < n and a[i]==0:
        i+=1
    return i

def isZero(A,N=False):
    '''Check if A modulo N = 0 for all values in A.'''
    if N:
        A = np.mod(A,N)
    return np.all(A == 0)

def binn(SX):
    '''Find the row length n of SX - could be None''' 
    if SX is None:
        return 0 
    SX = ZMat2D(SX) 
    m,n = np.shape(SX)
    return n

##########################
## Changing Shape of ZMat
##########################

def ZMat1D(A,n):
    '''Return a 1-dimensional integer numpy array from A of length n'''
    A = ZMat(A)
    if np.ndim(A) != 1:
        A = np.reshape(A,(-1,))
    if len(A) < n:
        A = np.tile(A,n//len(A))
    if len(A) > n:
        A = A[:n]
    return A

def ZMat2D(A):
    '''Return a 2-dimensional integer numpy array from A.'''
    A = ZMat(A)
    if np.ndim(A) == 2:
        return A
    if np.ndim(A) == 0:
        return ZMat([[A]])
    if np.ndim(A) == 1:
        return ZMat([A])    
    d = np.shape(A)[-1]
    return np.reshape(A,(-1,d))

def matResize(A,m,n,check=False):
    '''Resize integer matrix A to be m x n'''
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
    '''Convert ZMat to one-dimensional list of values.'''
    b = ZMat2D(b)
    return b.flatten().tolist()

##########################
## String I/0 for ZMat
##########################
def row2components(r):
    '''For integer vector r return indices for the non-zero values ix=supp(r) and the non-zero values r[ix].
    Useful for displaying large vectors.'''
    ix = ZMat(np.nonzero(r))
    return ix, r[ix]

def row2compStr(r):
    '''Display row r using indices for non-zero values and list of non-zero values
    Useful for displaying large vectors.'''
    ix,vals = row2components(r)
    ix = str(mat2list(ix))
    vals = str(mat2list(vals))
    return f'{ix}:= {vals}'.replace(" ","")

def ZMat2compStr(A):
    '''Display 2D integer matrix A using indices for non-zero values and list of non-zero values
    Useful for displaying large vectors.'''
    return "\n".join([row2compStr(r) for r in A])

def str2ZMat(mystr):
    '''Convert string of single digit numbers or multi digit numbers split by spaces to an integer array'''
    if mystr.find(" ") > 0:
        mystr = mystr.split()
    return ZMat([int(s) for s in mystr])

def str2ZMatdelim(S=''):
    '''Convert string with rows separated by \r, \n "," or ; to 2D integer array.'''
    sep=','
    for s in "\r\n;":
        S = S.replace(s,sep)
    S = S.split(sep)
    return ZMat([str2ZMat(s) for s in S])

def bin2Zmat(SX):
    '''Convert multiple types of binary vector input to integer matrix.
    SX is either string or array.'''
    if SX is None:
        return SX 
    ## convert string to ZMat
    if isinstance(SX,str):
        return str2ZMatdelim(SX)
    ## convert array to ZMat
    return ZMat(SX)

def ZMat2str(A,N=None):
    '''Return string version of integer matrix A.'''
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
    '''Join text vector a using sep - for display of ZMat.'''
    return sep.join(a)

def ZmatPrint(A,N=None):
    '''Print integer matrix A'''
    return "\n".join(ZMat2str(ZMat2D(A),N))

#################################################
## Debugging Functions
#################################################

def currTime():
    '''Return current time'''
    return time.process_time()

def startTimer():
    '''Start timer for algorithm and set global variable startTime to be the current time.'''
    global startTime
    startTime = currTime()
    return startTime

def elapsedTime():
    '''Return the time elapsed from last startTimer() call.'''
    global startTime
    return -startTime + startTimer()

def func_name():
    """Return the name of the current function - for debugging."""
    return sys._getframe(1).f_code.co_name

def typeName(val):
    '''Return the name of the type of val in text form.'''
    return type(val).__name__

#########################################
## getVal/setVal - store properties of objects where these are complex to compute
#########################################

def getVal(obj,label):
    '''Get a value indexed by label from obj. If not set, run "getlabel" function.'''
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
    '''Check if object has value corresponding to label set'''
    if not hasattr(obj,label):
        return False
    return getattr(obj,label) is not None

def setVal(obj,label,val=None):
    '''Set value corresponding to label to val.'''
    setattr(obj,label,val)
    return val

##################################
## Integer logarithms
##################################

def logCeil(x,N=2):
    '''Return min(t) where x <= N^t'''
    i = 0
    while x > 0:
        x = x // N
        i = i+1
    return i


def log2int(N):
    '''Find t such that N = 2**t or None otherwise'''
    t = logCeil(N-1,2) 
    return t if 2 ** t == N else None

