import sys; sys.path.insert(0, '..')
from myutils3_v2 import *
import myutils_cython

def my_operation_py(D,X,ZV):
    for i in range(D.shape[0]):
        D[i,:] = np.outer(X[i,:], ZV[i,:]).ravel() # kron() is slower


ra.seed(19)
nTry = 10
d = 128
r = 5
N = 3000
D = np.zeros((N, d*r))
D2 = np.zeros((N, d*r))
X = ra.randn(N, d)
ZV = ra.randn(N, r)

tic()
for i in range(nTry):
    myutils_cython.calcRowwiseKron(D, X, ZV)
printExpr('toc()')

tic()
for i in range(nTry):
    my_operation_py(D2,X,ZV)
printExpr('toc()')


