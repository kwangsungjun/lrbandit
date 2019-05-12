from myutils3_v2 import *
import cvxpy as cvx, scipy.linalg as sla;

def sdp(X):
    N, d = X.shape;
    lam = cvx.Variable(N);
    t = cvx.Variable();
    obj = cvx.Minimize(-t);
    cons = [];
    cons.append( X.T * cvx.diag(lam) * X >> t*np.eye(d) );
    cons.append( lam >= 0 )
    cons.append( cvx.sum(lam) == 1 )

    prob = cvx.Problem(obj, cons)
    prob.solve()

    mylam = np.array(lam.value);
    mylam = mylam*(mylam>0.0);
    mylam = mylam/mylam.sum()
    sidx = mylam.argsort()[::-1];

    subset = sidx[:d]
    inv_min_sval = 1 / sla.svdvals(X[subset,:]).min()
    #- division by zero could happen; that's fine
    return sidx[:d], inv_min_sval

def random(X, nTry):
    """
        try n times and take the best
    """
    N, d = X.shape;

    vList = []
    idxBest = None;
    vBest = np.inf;
    for iTry in range(nTry):
        idx = ra.permutation(N)[:d]
        v = 1.0 / sla.svdvals(X[idx,:]).min()
        if (v < vBest):
            vBest = v;
            idxBest = idx;

    inv_min_sval = vBest
    return idxBest, inv_min_sval;

def hybrid(X, nTry):
    """
       best of both sdp and random
    """
    subset, inv_min_sval = sdp(X);
    subset2, inv_min_sval2 = random(X, nTry)
    if inv_min_sval2 < inv_min_sval:
        subset = subset2
        inv_min_sval = inv_min_sval2
    return subset, inv_min_sval

if __name__ == "__main__":
    from tqdm import tqdm
    N = 100;
    d = 10;

    # try maybe 400 times and then take average.
    methodList = [sdp, lambda x: random(x, 1), lambda x: random(x, 5), lambda x: random(x, 10), lambda x: random(x, 20), lambda x: random(x, 20)];

    nTry = 400;
    V = np.zeros((nTry, len(methodList)))
    for iTry in tqdm(range(nTry)):
        X = ra.randn(N,d);
        X = X / np.sqrt((X**2).sum(1))[:,np.newaxis]
        for (j,f) in enumerate(methodList):
            [subset, _] = f(X);
            V[iTry,j] = 1 / ( sla.svdvals(X[subset,:])[-1] )

    V = np.array(V);
    printExpr('V.mean(0)')
    hybrid = np.min(V[:,[0,-1]],axis=1).mean()
    printExpr('hybrid')  

# conclusion
# 20 repetition seems to be right. also, combination of the sdp and the random seems to work well.
# result:
# In [49]: run test_cvx.py
# 100%|█████████████████████████████████████████████████| 400/400 [00:21<00:00, 18.34it/s]
# V.mean(0) =  array([19.09555422, 49.68627662,  7.77901072,  6.0818769 ,  5.13942173,
#         5.19037659])
# hybrid =  4.686361413828722
# 





