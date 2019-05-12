from myutils3_v2 import *
from tqdm import tqdm
import sys
import myutils_cython
import matrixrecovery

recoverMatrix = matrixrecovery.rankone

#@profile
def main():
    #ra.seed(923)
    nIter = 1000
    d = 128
    r = 5
#    N = 2*(2*128*5 + r**2)
    N = int(np.ceil(1*(2*128*5 + r**2)))
    #R = 0.01
    R = 0.1
    C = 0.1

    ra.seed(99)
    Th = ra.randn(d,r) @ ra.randn(r,d) #ra.randn(d,d)
    #Th = ra.randn(d,d)

    initU = 10*ra.randn(d,r)
    #initU = np.zeros((d,r))
    U = initU
    V = initU # just a placeholder

    #- generate data
    X = ra.randn(N, d)
    Z = ra.randn(N, d)
    y = np.sum((X@Th) * Z, 1) + R * ra.randn(N)

    U,V,out_nIter,stat = recoverMatrix(X,Z,y,r,R=R, C=C, maxIter=400, verbose=True)
    objs = stat['objs']

    hatTh = U@V.T
    import matplotlib.pyplot as plt

    plt.ion()
    plt.figure()
    plt.plot(range(1,1+out_nIter), objs)
    plt.title('objs')
    #plt.plot(range(1,1+nIter), objs)
    #plt.legend(['residuals', 'objs'])

    ipdb.set_trace()

if __name__ == "__main__":
    main()
