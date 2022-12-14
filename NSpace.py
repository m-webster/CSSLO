import numpy as np
from common import *

#######################################
#### General GCD/Modulo Arithmetic ####
#######################################

## return c such that a - bc < a
## or None if b == 0
def Quo(a,b,N,check=False):
    if check is None:
        return True
    if check is not False:
        r = (a - b * check)
        return r >= 0 and r < b
    a = a % N
    b = b % N
    if b == 0:
        return None
    return (a // b) % N

## return c such that bc = a mod N 
## or None if no such c exists
def Div(a,b,N,check=False):
    a = a % N
    b = b % N
    if check is None:
        return True
    if check is not False:
        g = np.gcd(b,N)
        if a % g > 0:
            return True
        return (b * check) % N == a
    if b == 0:
        return None
    g = np.gcd(b,N)
    if a % g == 0:
        r = a % b
        while r > 0:
            a += N
            r = a % b
        return a // b % N
    return None

## return g,s,t,u,v such that:
## as + bt = g where g = gcd(a,b)
## au + bv = 0
def Gcdex(a,b,N,check=False):
    if check is None:
        return True
    if check is not False:
        (g,s,t,u,v) = check
        result = (s * a + t * b) % N == g 
        result &= (u * a + v * b) % N == 0 
        result &= (s * v - t * u) % N == 1
        return result
    s = 0
    old_s = 1
    t = 1
    old_t = 0
    r = b
    old_r = a
    while r != 0:
        quotient = old_r // r
        (old_r, r) = (r, old_r - quotient * r)
        (old_s, s) = (s, old_s - quotient * s)
        (old_t, t) = (t, old_t - quotient * t)
    p = np.sign(t * old_s - s * old_t)
    u = p * s
    v = p * t
    g = old_r
    s = old_s
    t = old_t
    return(g,s,t,u,v)

# ## multiplicative inverse of a mod N
# def inverse(a,N,check=False):
#     if check is None:
#         return True
#     if check is not False:
#         return a * check % N == 1
#     a = a % N
#     if a == 0:
#         return None
#     t = 0
#     newt = 1
#     r = N
#     newr = a
#     while newr > 0:
#         quotient = r // newr
#         (t, newt) = (newt, t - quotient * newt) 
#         (r, newr) = (newr, r - quotient * newr)

#     if r > 1:
#         return None
#     if t < 0:
#         t = t + N
#     return t

## return u such that a u mod N = 0 
## return 0 if a is a unit
## return 1 if a == 0
def Ann(a,N,check=False):
    if check is None:
        return True
    if check is not False:
        return (N // np.gcd(a,N) - check) % N == 0
    a = a % N
    if a == 0:
        return 1
    u = N // np.gcd(a,N)
    return u % N

def Split(a,N,check=False):
    if check is None:
        return True
    if check is not False:
        primes = PrimeNumbers(N)
        for p in primes:
            if check % p == 0 and a % p ==0:
                return False
            if N // check % p == 0 and a % p !=0:
                return False
        return True
    if N ==0:
        return 0
    a = a % N
    if a == 0:
        return 1
    r = np.int(np.ceil(np.log2(np.log2(N)))) if N > 1 else 1
# print('N,r',N,r)
    for i in range(r):
        a = a*a % N
    return N // np.gcd(a,N)

def Factorize(a):
    f = []
    m = []
    for x in range(2,np.int(np.ceil(a ** 0.5)+1)):
        if a % x == 0:
            i = 0
            f.append(x)
            while a % x == 0:
                i += 1
                a = a // x
            m.append(i)
    return f,m

def Factors(a):
    f = set()
    for x in range(2,np.int(np.ceil(a ** 0.5)+1)):
        if a % x == 0:
            f.add(x)
            f.add(a //x)
    f.add(a)
    return sorted(f)

## generate list of prime numbers <= a
## used for checking Split function
def PrimeNumbers(a):
    if a < 2:
        return []
    f = [2]
    for x in range(1,a // 2):
        isPrime = True
        xi = 2 * x + 1
        i = 0
        maxi = np.int(np.ceil(xi ** 0.5))
        while isPrime and i < len(f):
            isPrime = xi % f[i] > 0
            i += 1
        if isPrime:
            f.append(xi)
    return f

def Stab(a,b,N,check=False):
    if check is not False:
        return (np.gcd(a + check * b,N) - np.gcd.reduce([a,b,N])) % N == 0
    a = a % N
    b = b % N
    g = np.gcd.reduce([a,b,N])
    c = Split(a//g,N//g)
    return c % N

# def Stabex(a,N,check=False):
#     a = makeMat(a)
#     if check is not False:
#         check = makeMat(check)
#         g = np.gcd.reduce(list(a) + [N])
#         temp = np.sum(a * check) % N
#         return g == np.gcd(temp,N)
#     if len(a) == 0:
#         return []
#     c = [1]
#     temp = a[0]
#     for i in range(1,len(a)):
#         c.append(Stab(temp,a[i],N))
#         temp = (temp + c[i] * a[i]) % N
#     return c

## return a unit c such that a c mod N is gcd(a,N)
def Unit(a,N,check=False):
    if check is None:
        return True
    if check is not False:
        return np.gcd(check,N) == 1 and (check * a - np.gcd(a,N)) % N == 0
    a = a % N
    if a == 0:
        return 1
    g = np.gcd(a,N)
    s = Div(g,a,N)
# print('g,s',g,s)
    if g == 1:
        return s
    d = Stab(s,N//g,N)
# print('d,N //g',d,N //g)
    c = (s + d * N // g) % N
    return c

#######################################
####          Testing Functions    ####
#######################################

def BinaryTest(f,N=False):
    if N is False:
        N = np.random.randint(2,20)
    print('Testing',f,'N=',N)
    temp = [['a','c','Check']]    
    for a in range(N):
        for b in range(N):
            check = f(a,b,N)
            r = f(a,b,N,check)
            print(ZMat2str([a,b,check,r]))
            if not r:
                temp.append([a,b,check,r])
    if len(temp) > 1:
        for r in temp:
            print(ZMat2str(r))
    else:
        print("All OK")
        
def UnaryTest(f,N=False):
    if N is False:
        N = np.random.randint(2,20)
    print('Testing',f,'N=',N)
    temp = [['a','c','Check']]    
    for a in range(N):
        check = f(a,N)
        r = f(a,N,check)
        print(ZMat2str([a,check,r]))
        if not r:
            temp.append([a,check,f])
    if len(temp) > 1:
        for r in temp:
            print(ZMat2str(r))
    else:
        print("All OK")

def TestRingOps(N=False):
    if N is False:
        N = np.random.randint(2,20)
    Binaries = [Quo,Div,Gcdex,Stab]
    for f in Binaries:
        BinaryTest(f,N)
    Unaries = [Ann,Split,Unit]
    for f in Unaries:
        UnaryTest(f,N)    

#######################################
####   Howellization of Matrixes   ####
#######################################


## perform row operation specified by opData on matrix A
def doOperation(A,opData,N=1):
    op, data = opData

    ## swap rows A[j], A[m]
    if op == 's':
        (j,m) = data
        # A[[j,m]] = A[[m,j]] 
        A[j],A[m] = A[m],A[j]

    ## eliminate rows j,m using GCDex
    if op == 'u':
        (j,m,s,t,u,v) = data
        B = ZMat2D([[s,t],[u,v]])
        R = ZMat2D([A[j],A[m]])
        C = matMul(B, R, N)
        A[j] = C[0]
        A[m] = C[1]

    ## replace A[m] with A[m] + c * A[j]
    if op == 'a':
        (j,m,c) = data
        A[m] = np.mod(A[m] + c * A[j],N)

    ## replace A[j] with c * A[j] 
    ## valid only where c is a unit
    if op == 'm':
        (j,c) = data
        A[j] = np.mod(c * A[j],N)

    ## add c*A[j] to end of matrix
    ## used when c is a zero divisor
    if op == 'n':
        (j,c) = data
        # A = np.vstack([A,np.mod(c * A[j],N)])
        A.append(np.mod(c * A[j],N))

    ## replace A[m] with c * A[j]
    # if op == 'r':
    #     (j,m,c) = data
    #     A[m] = np.mod(c * A[j],N)
    return A


## check that all rows of A are in <H>
def CheckHow(A,H,N):
     for a in A:
          r, u = matResidual(H,a,N)
          if np.sum(r) > 0:
               print(a, 'not in <H>')
               return False
     return True

## check function GetUK
def CheckUK(A,H,U,K,N):
     if len(A) > 0:
          ## Check that U @ A = H
          if not np.all(np.isclose(matMul(U,A,N),H)):
               print('U @ A != H')
               return False
          ## Check that K @ A = 0
          if not np.all(np.isclose(matMul(K,A,N),0)):
               print('K @ A != 0')
               return False
          ## Check U + K is full rank
          U = np.vstack([U,K])
          D = matMul(np.transpose(U),U,N) if len(U) > len(U[0]) else matMul(U,np.transpose(U),N)
          if np.isclose(np.linalg.det(D),0):
               print('U+K is not full rank')
               print(ZmatPrint(A,N))
               return False
     return True

## get U, K such that:
## H = U@A
## 0 = K@A
def GetUK(A,H,rowops,N):
     m = len(A)
     U = ZMatI(m)
     U = doOperations(U,rowops,N)
     ## split U and K
     k = len(H)
     U = ZMat(U,m)
     K = RemoveZeroRows(U[k:])
     U = U[:k]
     return U,K

def getKer(A,N):
    # startTimer()
    A = ZMat(A)
    H, rowops = How(A.T,N)
    U,K = GetUK(A.T,H,rowops,N)
    K,rowops = How(K,N)
    # print(func_name(),elapsedTime(),deep_getsizeof(K))
    return K
    # At = np.transpose(A)
    # nsp = NSpace(At,N)
    # return nsp.getVal('Kt')

def prepHowRes(A,z):
    B = np.vstack([z,A])
    B = np.hstack([ZMatZeros((len(B),1)),B])
    B[0,0] = 1 
    return B

## residue of z with respect to A modulo N
def HowRes(A,z,N,retro=False):
    # print('np.shape(A)',np.shape(A))
    # print('np.shape(z)',np.shape(z))
    B = prepHowRes(A,z)
    # print(func_name(), 'B')
    # print(ZmatPrint(B))
    H,rowops = How(B,N)
    # print('H')
    # print(ZmatPrint(H))
    if retro:
        return H[0,1:], rowops
    return H[0,1:]    

def doOperations(A,rowops,N):
    A = [a for a in A]
    for opData in rowops:
        A = doOperation(A, opData,N)
    return ZMat(A)

## returns Howell basis of A mod N plus row operations to convert to this form
def How(A,N,reduced=True):
     rowops = []
     if len(A) == 0:
         return A, rowops
    #  B = np.copy(A)
     B = [a for a in A]
     m,n = np.shape(A)
    #  print(func_name(),'m,n',m,n)
     r = 0
     ## c is the column of B we are currently looking at
     for c in range(n):
          ## find j such that B[j][c] > 0
          j = r
          while (j < m and B[j][c] == 0):
               j +=1
          if j < m:
               ## found j: if j > r, swap row j with row r
               if j > r:
                    rowops.append(('s',(j,r)))
                    B = doOperation(B,rowops[-1],N)
               ## Multiplying by x ensures that B[r][c] is a minimal representative
               x = Unit(B[r][c],N)
               if(x > 1):
                    rowops.append(('m',(r,x)))
                    B = doOperation(B,rowops[-1],N)            
               ## eliminate entries in column c below row r
               for j in range(r+1,m):
                    if B[j][c] % N > 0:
                         (g,s,t,u,v) = Gcdex(B[r][c],B[j][c],N)
                         rowops.append(('u',(r,j,s,t,u,v)))
                         B = doOperation(B,rowops[-1],N)  
               ## ensure entries in column c above row r are less than B[r][c]
               b = B[r][c]
               for j in range(r):
                    if B[j][c] >= b:
                         x = Quo(B[j][c],b,N)
                         if x is None:
                             print('Quo(B[j][c],b,N)',B[j][c],b,N,x)
                         rowops.append(('a',(r,j,-x)))
                         B = doOperation(B,rowops[-1],N)                
               ## Multiplying by x = Ann(b) eliminates b = B[r][c], but rest of the row may be non-zero
               ## If x > 0 then b is a zero divisor and we add a row
               ## If x == 0, b is a unit and we move to the next value of l
               x = Ann(b,N)
               if x > 0:
                    rowops.append(('n',(r,x)))
                    B = doOperation(B,rowops[-1],N)  
                    m = len(B)
               r +=1
    #  H = RemoveZeroRows(B) 
    #  print(func_name(),len(B))
     H = RemoveZeroRows(ZMat(B,n))
     return H,rowops


## returns Strong lower triangular form of binary matrix 
def Triag(A):
    m,n = np.shape(A)
    if m == 0:
        return A
    H = ZMat([a for a in A])
    comp = []
    K = ZMatI(n)
    for j in range(n):
        i=0
        while (i < m and H[i][j] == 0):
            i +=1
        if i < m:
            comp.append(j)
            for l in range(j+1, n):
                if H[i][l] != 0:
                    H[:,l] = np.mod(H[:,j] + H[:,l],2)
                    K[:,l] = np.mod(K[:,j] + K[:,l],2)
    pivot = sorted(set(range(n)) - set(comp))
    H = H[:,comp]
    K = K[:,pivot]
    F = ZMatI(n)[:,pivot]
    return H, K, F


#######################################
####    Linear Algebra Modulo N    ####
#######################################


## main class of this module for linear algebra modulo N
class NSpace:

    def __init__(self,A,N):
        self.A = ZMat2D(A)
        self.At = np.transpose(A)
        self.N = N
        self.m = len(A)
        self.n = len(A[0]) if self.m > 0 else 0  
        self.getVal('H')   

    def getVal(self,a):
        if hasattr(self, a):
            return getattr(self,a)
        if a == 'H':
            self.H,self.opsH = How(self.A,self.N)
        if a in ['P','Kt']:
            self.P, self.Kt = GetUK(self.A,self.H,self.opsH,self.N)
        if a == 'T':
            self.T,self.opsT = How(self.At,self.N)
        if a in ['K','S']:
            self.getVal('T')
            self.S, self.K = GetUK(self.At,self.T,self.opsT,self.N)

        return getattr(self,a)
        
    def simplifyKer(self):
        self.getVal('K')
        self.K,ops = How(self.K,self.N)
        if isZero(self.K):
            self.K = ZMatZeros((1,self.n))                

    ## solve Ax = b modulo N
    def makeOffset(self,b,check=False):
        self.getVal('S')
        if check is not False:
            v,d = check
            return matEqual(d,self.checkOffset(b,v),self.N)
        ## check if b is in the span of T
        r,u = matResidual(self.T,b,self.N)
        if not isZero(r):
            print("makeOffset: not in span residual",r)
        c = matLinearComb(self.S,u,self.N)
        c = matResize(c,1,self.n)
        ## r and d should be all zero if exact solution exists
        ## otherwise, difference between estimate and solution
        d = self.checkOffset(b,c)
        # report(func_name(),'d',d)
        return c[0],d[0]
    
    def checkOffset(self,b,c):
        ## result is all zero if c is an exact solution
        ## otherwise, difference between estimate and solution
        b = colVector(b)
        c = colVector(c)
        v = matMul(self.A,c,self.N)
        return rowVector(matAdd(b, -v,self.N))


def matResidual(A,b,N,check=False):
    ## take vector b
    ## return resdidual after subtracting Howell basis vectors
    ## and vector with multiples of each basis vector
    # b1,a1 = matResidue(A,b,N)
    # print(func_name(),A)
    b = rowVector(b)
    A = ZMat2D(A)
    if check is not False:
        x, o = check
        return matEqual(b,matLinearComb(A,x,N,o))
    b = makeList(np.mod(b,N))
    if len(A) == 0 or len(A[0]) == 0:
        return b,ZMat2D([])
    temp = []
    for i in range(len(A)):
        ix = Highbit(A[i])
        c = 0
        if ix >= 0:
            d = Quo(b[ix],A[i][ix],N)
            c = d if d else 0
        temp.append(c)
        b = np.mod(b - c * A[i],N)
    a = ZMat(temp)
    return b,a

def matLinearComb(A,b,N,o=[0],check=False):
    ## take vector b representing pattern of basis vectors
    ## return linear combination of basis + offset o
    if check is not False:
        b1,o1 = matResidual(A,check,N)
        return matEqual(b,b1) and matEqual(o,o1)
    o = rowVector(o)
    b =  rowVector(matMul(b, A, N))
    return np.mod(b + o, N)

## given a set of matrix A in Howell form
## Modulo N
## Produce the set of generators with the lowest weight up to the nth entry
# def lowWeightGenerators(A,N=2,n=None,check=False):
#     if n is None:
#         m,n = np.shape(A)
#     if check is not False:
#         S,H = check
#         # print('Checking',func_name())
#         # print('H')
#         # print(ZmatPrint(H))
#         for x in A:
#             r,u = matResidual(H,x[:n],2)
#             if sum(r) > 0:
#                 return False 
#         return True

#     temp = dict()
#     for x in RowSpan(A,N):
#         w = weight(x[:n])
#         if w > 0:
#             if w not in temp:
#                 temp[w] = []
#             temp[w].append(x)
#     S,H = [],[]
#     for w in sorted(temp.keys()):
#         print(func_name(), w,len(temp[w]))
#         for x in temp[w]:
#             if len(S) == 0:
#                 S.append(x)
#                 H.append(x[:n])
#             else:
#                 r, u = matResidual(H,x[:n],N)
#                 if sum(r)> 0: 
#                     H.append(r)
#                     S.append(x)
#     return  ZMat(S),ZMat(H)
    



##################################################
####          Intersection of Spans           ####
##################################################

## Calculate intersection of two affine spaces
## U1 = o1 + <A1> mod N
## U2 = o2 + <A2> mod N
def affineIntersection(A1,o1,A2,o2,N,C=False):
    if C is not False:
        A,o = C
        tocheck = np.mod(o + A,N)
        tocheck = np.vstack([[o],tocheck])
        for v in tocheck:
            v1 = np.mod(v - o1,N)
            b,u = matResidual(A1,v1,N)
            if not isZero(b):
                print(v1, 'Not in span of A1')
                return False
            v2 = np.mod(v - o2,N)
            b,u = matResidual(A2,v2,N)
            if not isZero(b):
                print(v2, 'Not in span of A2')
                return False
        return True

    nsp = NSpace(np.vstack([A1,A2]),N)
    # nsp.simplifyH()
    v = np.mod(o1-o2,N)
    b,u = matResidual(nsp.H,v,N)
    if not isZero(b):
        ## there is no solution o1-o2 = - v1@A1 + v2@A2 <=> o1 + v1@A1 = o2 + v2@A2
        return False
    ## residue of o1-o2 wrt A1 = v2@A2
    b,u = matResidual(A1,v,N)
    ## new offset is o2 + v2@A2
    o = np.mod(b + o2,N)
    ## new affine space is intersection
    A = nsIntersection([A1,A2],N)
    return A,o

## Calculate intersection of two affine spaces
## U1 = o1 + <A1> mod N
## U2 = o2 + <A2> mod N
## return o which is in both U1 and U2 or False
def affineIntercept(A1,o1,A2,o2,N,C=False):
    nsp = NSpace(np.vstack([A1,A2]),N)
    v = np.mod(o1-o2,N)
    b,u = matResidual(nsp.H,v,N)
    if not isZero(b):
        ## there is no solution o1-o2 = - v1@A1 + v2@A2 <=> o1 + v1@A1 = o2 + v2@A2
        return False
    ## residue of o1-o2 wrt A1 = v2@A2
    b,u = matResidual(A1,v,N)
    ## new offset is o2 + v2@A2
    o = np.mod(b + o2,N)
    return o

## intersection of multiple N spaces
def nsIntersection(Alist,N):
    if len(Alist) == 0:
        return False
    A = Alist[0]
    for B in Alist[1:]:
        AB = np.vstack([A,B])
        nsp = NSpace(AB,N) 
        nsp.getVal('Kt')
        C = matMul(nsp.Kt,np.vstack([A,np.zeros(B.shape,dtype=int)]),N)
        nsp = NSpace(C,N)
        A = nsp.H
    return A

## union of multiple N spaces
def nsUnion(Alist,N):
    if len(Alist) == 0:
        return False
    nsp = NSpace(np.vstack(Alist),N) 
    return nsp.H
