#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

def calcRowwiseKron(D, X, ZV):
    """
       X: N by d
       ZV: N by r
       returns D where D[i,:] = np.outer(X[i,:], ZV[i,:]).ravel()
    """
    cdef int N = D.shape[0]
    cdef int d = X.shape[1]
    cdef int r = ZV.shape[1]
    cdef double [:,:] vD = D # 'v' for view
    cdef double [:,:] vX = X
    cdef double [:,:] vZV = ZV
    cdef int cnt, i, j, k
    
    for i in range(N):
        cnt = 0
        for j in range(d):
            for k in range(r):
                vD[i,cnt] = vX[i,j] * vZV[i,k]
                cnt += 1
    return 


