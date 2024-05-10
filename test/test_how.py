from add_parent_dir import *
from common import *
from NHow import *
from CSSLO import *
import itertools as iter


#####################################################
## Testing functions
#####################################################

def testHow(A,N,tB=1):
    '''Test calculation of Howell form and Kernel of A modulo N'''
    A = ZMat(A)
    m,n = A.shape
    print('A')
    print(ZMatPrint(ZMatBlockSum(A,tB=tB,N=N)))
    H,U = getHU(A,N,tB=tB)
    print('H')
    print(ZMatPrint(ZMatBlockSum(H,tB=tB,N=N)))
    print('U')
    print(ZMatPrint(U))
    UA = matMul(U, A, N)
    print(f'Check H = U @ A {isZero(H - UA, N)}\n')
    K = getK(A,N,tB=tB)
    print('K')
    print(ZMatPrint(ZMatBlockSum(K,tB=tB,N=N)))
    KHt = matMul(K, H.T, N)
    print(f'Check K @ H.T = 0 {isZero(KHt)}\n')
    B = randomNMat(m,n,N)
    R,V,H,U,K = solveHU(A,B,N,tB=tB)
    print('Test HowSolve: B = R + (V + <K>) @ A mod N')
    print('H')
    print(ZMatPrint(ZMatBlockSum(H,tB=tB,N=N)))
    print('R')
    print(ZMatPrint(ZMatBlockSum(R,tB=tB,N=N)))
    print('V')
    print(ZMatPrint(V))
    print(f'Check B = R + V @ A mod N: {isZero(R + matMul(V, A,N) - B, N)}')
    print(f'Check K @ A mod N = 0: {isZero(matMul(K, A, N))}')
    print(f'Check U @ A mod N = H: {isZero(H - matMul(U, A, N))}')

def testRingFunctions(N):
    '''Test ring functions modulo N by producing tables'''
    ## Unary functions
    uFunc = [Ann_jit, Unit_jit, Split_jit]
    for myfunc in uFunc:
        print(myfunc.__name__)
        print(np.array([myfunc(i,N) for i in range(N)],dtype=np.int16))
    ## Binary functions
    bFunc = [Stab_jit,Div_jit,Gcdex_jit]
    for myfunc in bFunc:
        print(myfunc.__name__)
        print(np.array([[myfunc(i,j,N) for i in range(N)] for j in range(N)],dtype=np.int16))

def GF2RREF(A,N):
    '''Calculate RREF modulo 2 using Galois package'''
    import galois
    A = galois.GF2(A)
    return A.row_reduce()

def ldpcRREF(A,N):
    import ldpc.mod2
    return ldpc.mod2.reduced_row_echelon(A)

def testHowRandom(m=10,n=15,N=4,nBlocks=1):
    '''Generate a random matrix and test Howell functions'''
    print("########################################################")
    print(f'Test Howell operations on random matrix m,n={m,n},N={N}')
    print("########################################################")
    A = randomNMat(m,n * nBlocks,N)
    testHow(A,N,nBlocks)

def randomNMat(m,n,N):
    '''Generate random matrix modulo N'''
    mytype = np.int8 if N==2 else np.int16
    return np.random.randint(N,size=(m,n),dtype=mytype)

def testHowResources(m=1200,n=1600,N=2):
    '''Test memory and time usage for Howell form'''
    print("########################################################")
    print(f'Test memory and time usage for Howell form m,n={m,n},N={N}')
    print("########################################################")
    tracemalloc.start()
    ## default function from NHow
    testFun = 'getH'
    ## Joschke Roffe LDPC function - mod 2 only
    # testFun = 'ldpcRREF'
    ## Galois row_reduce function - mod 2 only
    # testFun = 'GF2RREF'
    # testFun = 'How'
    cProfile.run(f'{testFun}(randomNMat({m},{n},{N}),{N})')
    # displaying the memory
    
    print('Memory usage: current/peak',tracemalloc.get_traced_memory())
    # stopping the library
    tracemalloc.stop()

def testHowStorjohann():
    '''Examples from Arne Storjohann's work'''
    print("########################################################")
    print("Examples from Arne Storjohann's work")
    print("########################################################")
    i = 0
    ## Examples from Storjohann and Mulders, Fast Algorithms for Linear Algebra Modulo N
    ## the following 4 matrices have the same Howell matrix form:

    N = 12
    A = ZMat([[8,5,5],[0,9,8],[0,0,10]])
    i += 1
    print(f'\n\nTEST {i}: N={N}')
    testHow(A,N)

    N = 12
    A = ZMat([[4,1,10],[0,0,5]])
    i += 1
    print(f'\n\nTEST {i}: N={N}')
    testHow(A,N)

    N = 12
    A = ZMat([[4,1,0],[0,0,1]])
    i += 1
    print(f'\n\nTEST {i}: N={N}')
    testHow(A,N)

    N = 12
    A = ZMat([[4,1,0],[0,3,0],[0,0,1]])
    i += 1
    print(f'\n\nTEST {i}: N={N}')
    testHow(A,N)

    # Example from Storjohann's PhD Dissertation, Page 7 - both have the same Howell Form
    N = 16
    A = ZMat([[8,12,14,7]])
    i += 1
    print(f'\n\nTEST {i}: N={N}')
    testHow(A,N)

    N = 16
    A = ZMat([[8,12,14,7],[8,4,10,13]])
    i += 1
    print(f'\n\nTEST {i}: N={N}')
    testHow(A,N)

    # Example from Storjohann's PhD Dissertation, Page 26
    N = 4
    A = ZMat([[2,3,2,2],[0,0,3,3],[2,3,2,2]])
    i += 1
    print(f'\n\nTEST {i}: N={N}')
    testHow(A,N)

    # Typical example eg when calculating Canonical Generators
    # When N is prime, same result as Gaussian Elimination
    N = 2
    A = [[0,0,0,1,0,0,0,0,0,0],
         [0,0,0,1,0,0,1,1,1,0],
         [0,0,0,1,0,1,0,1,0,1],
         [0,0,0,1,0,1,1,0,1,1],
         [0,0,1,0,1,0,0,0,1,1],
         [0,0,1,0,1,0,1,1,0,1],
         [0,0,1,0,1,1,0,1,1,0],
         [0,0,1,0,1,1,1,0,0,0],
         [0,1,0,0,0,0,0,1,0,0],
         [0,1,0,0,0,0,1,0,1,0],
         [0,1,0,0,0,1,0,0,0,1],
         [0,1,0,0,0,1,1,1,1,1],
         [1,0,0,0,1,0,0,1,1,1],
         [1,0,0,0,1,0,1,0,0,1],
         [1,0,0,0,1,1,0,0,1,0],
         [1,0,0,0,1,1,1,1,0,0]]

    i += 1
    print(f'\n\nTEST {i}: N={N}')
    testHow(A,N)


def testSpan(m=10,n=15,N=4):
    print(f'Checking Span operations: m,n = {m},{n}; N = {N}')
    A = randomNMat(m,n,N)
    oA = randomNMat(1,n,N)
    B = randomNMat(m,n,N)
    oB = randomNMat(1,n,N)
    C = nsUnion([A,B],N)
    print(f'Checking nsUnion Rank={len(C)}',nsUnion([A,B],N,C))
    C = nsIntersection([A,B],N)
    if C is False:
        print('Intersection Empty')
    else:
        print(f'Checking nsIntersection Rank={len(C)}',nsIntersection([A,B],N,C))
    C = affineIntercept(A,oA,B,oB,N)
    if C is False:
        print('affineIntercept Empty')
    else:
        print(f'Checking affineIntercept {C}',affineIntercept(A,oA,B,oB,N,C))    
    C = affineIntersection(A,oA,B,oB,N)
    if C is False:
        print('affineIntersection Empty')
    else:
        AI, oAI = C
        print(f'Checking affineIntersection Rank={len(AI)}',affineIntersection(A,oA,B,oB,N,C))

# testHowRandom(m=10,n=15,N=2,nBlocks=2)
testHowResources(m=1200,n=1600,N=2)
# testSpan(m=8,n=15,N=4)
# testHowStorjohann()