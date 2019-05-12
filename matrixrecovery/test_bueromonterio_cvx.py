"""
Extremely slow!!
"""
from myutils3_v2 import *
from tqdm import tqdm
import cvxpy as cvx


#@profile
def main():
    #ra.seed(923)
    nIter = 1000
    d = 90 #128
    r = 1   #5
#    N = 2*(2*128*5 + r**2)
    N = 40
    #R = 0.01
    R = 0.1
    C = 0.1

    Th = ra.randn(d,r) @ ra.randn(r,d) #ra.randn(d,d)
    #Th = ra.randn(d,d)

    initU = 10*ra.randn(d,r)
#     #initU = np.zeros((d,r))
#     U = initU
#     V = initU # just a placeholder
    varU = cvx.Variable((d,r))
    varV = cvx.Variable((d,r))
    varU.value = initU
    varV.value = initU # just a placeholder

    #- generate data
    X = ra.randn(N, d)
    Z = ra.randn(N, d)
    y = np.sum((X@Th) * Z, 1) + R * ra.randn(N)
    #y = np.sum((X@Th) * Z, 1) 

    residuals = []
    objs = []
    myeye = R*C*np.eye(d*r) 
    for iIter in tqdm(range(1,1+nIter)):
        D = np.zeros((N,d*r))


        if iIter % 2 == 0:
            varU = cvx.Variable((d,r))
            V = varV.value
            obj = cvx.Minimize(cvx.norm( cvx.sum(cvx.multiply(X@(varU @ V.T),Z), 1) - y, 2)**2 + (C*R)*cvx.norm( varU ) ** 2)
            prob = cvx.Problem(obj, [])
            ipdb.set_trace()
            prob.solve()
        else:
            varV = cvx.Variable((d,r))
            U = varU.value
            obj = cvx.Minimize(cvx.norm( cvx.sum(cvx.multiply(X@(U @ varV.T),Z), 1) - y, 2)**2 + (C*R)*cvx.norm( varV ) ** 2)
            prob = cvx.Problem(obj, [])
            ipdb.set_trace()
            prob.solve()

        #- compute residual
        hatTh = U.value @ V.value.T
        obj = la.norm(((X@hatTh)*Z).sum(1) - y, 2)**2 + R*C*(la.norm(U, 'fro')**2 + la.norm(V, 'fro')**2)
        residuals.append( la.norm(hatTh - Th, 'fro') )
        objs.append( obj )
        if (iIter >=2 and iIter % 2 == 0):
    #        ipdb.set_trace()
            # TODO let's see if I just improved anything...
            pass
        pass

    residuals = np.array(residuals)

    yhat = ((X@hatTh)*Z).sum(1)
    import matplotlib.pyplot as plt

    plt.ion()
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(range(1,1+nIter), residuals)
    plt.title('residuals')
    plt.subplot(1,2,2)
    plt.plot(range(1,1+nIter), objs)
    plt.title('objs')
    #plt.plot(range(1,1+nIter), objs)
    #plt.legend(['residuals', 'objs'])

main()

