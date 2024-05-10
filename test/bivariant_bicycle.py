
from add_parent_dir import *
from common import *
from NHow import *
from CSSLO import *
import itertools as iter
import numpy as np

def SMatrix(n):
    S = ZMatI(n)
    return np.roll(S,1,axis=-1)

def matPower(A,p):
    return np.linalg.matrix_power(A,p)

def BivariantBicycle(l,m,Apoly,Bpoly):
    Sl = SMatrix(l)
    Sm = SMatrix(m)
    x = np.kron(Sl,ZMatI(m))
    y = np.kron(ZMatI(l),Sm)
    MList = [matPower(m,p) for (m,p) in zip([x,y,y],Apoly) ]
    A = np.mod(np.sum(MList,axis=0),2)
    MList = [matPower(m,p) for (m,p) in zip([y,x,x],Bpoly) ]
    B = np.mod(np.sum(MList,axis=0),2)
    SX = np.hstack([A,B])
    SZ = np.hstack([B.T,A.T])
    return SX,SZ

# 72-qubit code d=6
# l,m = 6,6
# Apoly = [3,1,2]
# Bpoly = [3,1,2]


# 90-qubit code d=10
# l,m = 15,3
# Apoly = [9,1,2]
# Bpoly = [0,2,7]

# 108-qubit code d=10
# l,m = 9,6
# Apoly = [3,1,2]
# Bpoly = [3,1,2]

# # # ## 144-qubit code d=12
# l,m = 12,6
# Apoly = [3,1,2]
# Bpoly = [3,1,2]

# # # # 288-qubit code d=18
# l,m = 12,12
# Apoly = [3,2,7]
# Bpoly = [3,1,2]

# # # 360-qubit code - d < 24
l,m = 30,6
Apoly = [9,1,2]
Bpoly = [3,25,26]

# # # # 756-qubit code - d < 34
# l,m = 21,18
# Apoly = [3,10,17]
# Bpoly = [5,3,19]

startTimer()
SX,SZ = BivariantBicycle(l,m,Apoly,Bpoly)
C = matMul(SX,SZ.T,2)
print(f'Stabilisers commute:',np.sum(C) == 0,f"{elapsedTime():.3f}s" )
SX,LX,SZ,LZ = CSSCode(SX,SZ=SZ)
k,n = np.shape(LX)
print(f'[[{n},{k}]] Code',f"{elapsedTime():.3f}s")
LX = minWeightLZ(SX,LX)
print('LX weights:',freqTablePrint(np.sum(LX,axis=-1)),f"{elapsedTime():.3f}s")