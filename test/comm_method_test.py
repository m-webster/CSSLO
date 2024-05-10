from add_parent_dir import *
from common import *
from CSSLO import *
import itertools as iter

def comm_method_rept():
    ########################################################
    ## Default Values
    ########################################################
    SX,LX,SZ,LZ = None,None,None,None
    t=2

    ###########################################################
    ## Hyperbolic Tesselations
    ###########################################################

    ## Hyperbolic Quantum Colour Codes https://doi.org/10.48550/arXiv.1804.06382
    ## 2D Colour Codes
    # myfile = "RG-3-6.txt"
    # myfile = "RG-3-8.txt"
    # myfile = "RG-3-10.txt"
    # myfile = "RG-3-12.txt"
    # myfile = "RG-3-14.txt"

    ## 3D Colour Codes - Flag complexes
    ## tesselation of 24-cells
    # myfile = "FG-3-3-3-3.txt"
    # ## tesselation of 48-cells
    # myfile = "FG-3-4-3-4.txt"
    # myfile = "FG-4-3-4-2.txt"

    # ## Other Hyperbolic Tesselations
    # ## Hyperbolic and Semi-Hyperbolic Surface Codes for Quantum Storage  https://doi.org/10.1088/2058-9565/aa7d3b
    # myfile = "RG-4-5.txt"
    # myfile = "RG-5-5-codes.txt"
    # # myfile = "RG-7-7-codes.txt"
    # # myfile = "RG-13-13-codes.txt"

    ## 3D Tesselations
    ## dodecahedra i=3
    # myfile = "RG-5-3-5-2.txt"
    ## icosahedra - i=0
    myfile = "RG-3-5-3-2.txt"
    ## cubes -i=4
    # myfile = "RG-5-3-4-2.txt"
    ## cubes - Euclidean
    # myfile = "RG-4-3-4-2.txt"

    codeList = importRGList(myfile)
    print(printRGList(codeList,myfile,checkValid=False))
    # ## pick the code by varying ix
    ix = 0
    myrow = codeList[ix]

    # for myrow in codeList:
    #     print(analyseCode(myrow))


    ## Print cellulation for presentation graphically
    # A = hypCellulation(myrow)
    # colouring = NColour(A,3)
    # print('3-Colourable', testColouring(A,colouring))

    # ## Colour Code
    SX, SZ = complex2ColourCode(myrow[1])

    # print('tOrthogonal',tOrthogonal(SX))

    #########################################################################
    ## Poset Codes from Self-Orthogonal Codes Constructed from Posets and Their Applications in Quantum Communication https://doi.org/10.3390/math8091495
    #########################################################################

    # Form 1: Quantum Hamming codes
    # a1 >= 3; a2 = b1 = b2 = 0
    # a1,a2,b1,b2 = 5,0,0,0

    # ## Form 2: a1 > a2 >= 3; b1 = b2 = 0
    # # a1,a2,b1,b2 = 5,3,0,0

    # ## Form 3: a1 = 1; b1 >= 3; a2 = b2 =0
    # # a1,a2,b1,b2 = 1,0,4,0

    # # ## Form 4: a1 = a2 = 1; b1 > b2 >= 3
    # # a1,a2,b1,b2 = 1,1,5,3

    # SX = posetCode(a1,a2,b1,b2)
    # SZ = SX

    ###################################################
    ###################################################

    startTimer()

    # print("\nCalculating Logical X and Z")
    SX,LX,SZ,LZ = CSSCode(SX,LX,SZ,LZ,simplifyGens=False)
    r,n = np.shape(SX)
    k,n = np.shape(LX)
    print(f'[[{n},{k}]] Code; elapsedTime:{elapsedTime()}')

    N = 1 << t

    compact = n > 8

    # print('CSS Code Checks and Logicals:')
    # print_SXLX(SX,LX,SZ,LZ)
    print('SX',freqTable(np.sum(SX,axis=-1)))
    print('LX',freqTable(np.sum(LX,axis=-1)))
    # print('SZ',freqTable(np.sum(SZ,axis=-1)))
    # print('LZ',freqTable(np.sum(LZ,axis=-1)))

    # print('\nCalculating Distance')
    # LX = minWeightLZ(SX,LX)
    # LZ = minWeightLZ(SZ,LZ)
    # dX = 0 if len(LX) == 0 else min(np.sum(LX, axis=-1))
    # dZ = 0 if len(LZ) == 0 else min(np.sum(LZ, axis=-1))
    # print(f'dX={dX}, dZ={dZ}; elapsedTime:{elapsedTime()}')

    
    r,n = np.shape(SX)
    N = 1 << t
    ## Logical identities
    if t == 1: 
        K_M = ZMatZeros((1,n+1))
    elif t == 2 and SZ is not None:
        K_M = np.hstack([SZ,ZMatZeros((len(SZ),1))]) * 2
    else:
        K_M = LIAlgorithm(LX,SX,N//2,compact,debug=False) * 2

    ## new method
    LZ = DiagLOComm_new(SX,K_M,N)
    # LZ = DiagLOComm(SX,K_M,N)
    print(f'len(LZ): {len(LZ)}; elapsedTime:{elapsedTime()}')
    # print('LZ')
    # print(ZMatPrint(LZ))

    # LZ = DiagLOComm(SX,K_M,N)
    # print(f'len(LZ): {len(LZ)}; elapsedTime:{elapsedTime()}')
    # print('LZ')
    # print(ZMatPrint(LZ))

    # print('\nCalculating Transversal Logical Operators')
    # zList, qList, V, K_M = comm_method(SX, LX, SZ, t,compact,debug=False)

    # print(f'(action : z-component); elapsedTime:{elapsedTime()}')
    # for z, q in zip(zList,qList):
    #     print( CP2Str(2*q,V,N))


# def DiagLOComm_new(SX,K_M,N):
#     '''Return z components of generating set of logical XP operators using Commutator Method.
#     Inputs:
#     SX: X-checks
#     K_M: phase and z components of diagonal logical XP identities 
#     N: required precision
#     Output:
#     LZ: list of z components of logical XP operators.'''
#     # print('K_M')
#     # print(ZMatPrint(K_M))
#     r,n = SX.shape
#     LZ = None
#     SXParts = SXPartition(SX)
#     print('SX Partitions',len(SXParts))
#     LZ = None
#     for XList in SXParts:
#         w = np.sum(XList,0)
#         ix = [i for i in range(n) if w[i] == 0]
#         RX = ZMatZeros((len(ix),n))
#         for i in range(len(ix)):
#             RX[i,ix[i]] = 1
#         XConstr = [RX]
#         # for x in XList:
#         #     ix += bin2Set(x)
#         # K = getH(K_M[:,ix],N)
#         # print('test K')
#         # print(ZMatPrint(K))
#         for x in XList:
#             Rx,ix = commZ_new(x,K_M,N)
#             # print(ix)
#             RX = ZMatZeros((len(Rx),n))
#             RX[:,ix] = Rx
#             XConstr.append(RX)
#         XConstr = np.vstack(XConstr)
#         # print('XConstr')
#         # print(ZMatPrint(XConstr))
#         LZ = XConstr if LZ is None else nsIntersection([LZ,XConstr],N)
#     return LZ

# def commZ_new(x,K_M,N):
#     '''Return generating set of Z-components for which group commutator with X-check x is a logical identity.
#     Inputs:
#     x: an X-check (binary vector of length n)
#     K_M: phase and z components of diagonal logical XP identities 
#     N: required precision
#     '''
#     n = len(x)
#     ## get set bits
#     xSet = bin2Set(x)
#     ## number of set bits
#     wx = len(xSet)
#     ## reorder K_M so that set bits are to the right, then phase column
#     ix = [i for i in range(n) if i not in xSet] + xSet + [n]
#     LI = K_M[:,ix]
#     ## How preferentially eliminates entries to the left
#     LI = getH(LI,N)
#     ## get rows of LI which are all zero to the left of set bits
#     w = np.sum(LI[:,:n - wx],axis=-1)
#     LI = LI[w==0,n-wx:]
#     ## rows are of form (-2 I | 1) because we require phase component equal to - x.z 
#     Rx = np.hstack([(N-2) * ZMatI(wx),np.ones((wx,1),dtype=int)])
#     LI = nsIntersection([LI,Rx],N)
#     # adjustment to ensure x.z = phase (first col)
#     phaseAdj = np.sum(LI[:,:-1],axis=-1) - 2 * LI[:,-1]
#     LI[:,0] = np.mod(LI[:,0] + phaseAdj,2*N)
#     ## solutions z are half the values in the intersection
#     LI = LI[:,:-1] // 2
#     N2 = np.hstack([ZMatI(wx-1),np.ones((wx-1,1),dtype=int)])
#     LI = np.vstack([LI,N2 * (N // 2)])
#     return LI, xSet



# comm_method_rept()
# exit()

print("########################################################")
print(f'Test memory and time usage for commutator method')
print("########################################################")
tracemalloc.start()
cProfile.run(f'comm_method_rept()')
# displaying the memory
print('Memory usage: current/peak',tracemalloc.get_traced_memory())
# stopping the library
tracemalloc.stop()