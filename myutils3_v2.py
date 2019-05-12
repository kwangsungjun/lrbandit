#-------------------------------------------------------------------------
# KJUN
from numpy import *
import gzip, numpy, numpy as np, scipy.linalg as sla, os, operator, traceback, sys, ipdb
import numpy.random as ra, ipdb, pickle as pickle, time, numpy.linalg as la, scipy.stats as st;
import itertools          # For Multiprobe
import copy
from datetime import datetime
from collections import OrderedDict
from types import SimpleNamespace
#import iprod
#from distcolors import get_distinguishable_colors

np.set_printoptions(precision=4);

################################################################################
# Pickle
################################################################################
def LoadPickle(fName):
    """ load a pickle file. Assumes that it has one dictionary object that points to
 many other variables."""
    if type(fName) == str:
        try:
            fp = open(fName, 'rb')
        except:
            print("Couldn't open %s" % (fName))
            traceback.print_exc(file=sys.stderr)
            ipdb.set_trace();
    else:
        fp = fName
    try:
        ind = pickle.load(fp)
        fp.close()
        return ind
    except:
        print("Couldn't read the pickle file", fName)
        traceback.print_exc(file=sys.stderr)
        ipdb.set_trace();

def SavePickle(filename, var, protocol=2):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(var, f, protocol=protocol)
        statinfo = os.stat(filename,)
        if statinfo:
            print("Wrote out", statinfo.st_size, "bytes to", \
                filename)
    except:
        print("Couldn't pickle the file", filename)
        traceback.print_exc(file=sys.stderr)
        ipdb.set_trace();

def LoadPickleGzip(fName):
    if type(fName) == str:
        try:
            fp = gzip.GzipFile(fName, 'rb')
        except:
            print("Couldn't open %s" % (fName))
            return None
    else:
        fp = fName
    try:
        ind = pickle.load(fp)
        fp.close()
        return ind
    except:
        print("Couldn't read pickle file", fName)
        traceback.print_exc(file=sys.stderr)

def SavePickleGzip(filename, var, protocol=2):
    try:
        with gzip.GzipFile(filename, 'wb') as f:
            pickle.dump(var, f, protocol=protocol)
        statinfo = os.stat(filename,)
        if statinfo:
            print("Wrote out", statinfo.st_size, "bytes to", \
                filename)
    except:
        print("Couldn't pickle index to file", filename)
        traceback.print_exc(file=sys.stderr)


# saves variables in 'varList' to filename, and it searches variables from given 'dic'
# typically, if you want to save variables from current python context
# set 'locals()' as parameter 'dic'
def savePickleFromDic(varList, fileName, dic):
    varDic = {};
    for k in varList:
        varDic[k] = dic[k];
    f = open(fileName, 'wb');
    pickle.dump(varDic, f);
    f.close();

################################################################################
# other utility function
################################################################################

def SaveToDict(dic, varList):
    varDic = {};
    for k in varList:
        varDic[k] = dic[k];
    return varDic;

from types import SimpleNamespace
def SaveToSimpleNamespace(dic, varList):
    ret = SimpleNamespace()
    for k in varList:
        varDic[k] = dic[k];
    return varDic;

    import inspect
    frame = inspect.currentframe()
    try:
        loc = frame.f_back.f_locals
        print(expr, '= ', end=' ') 
        if (bPretty):
            pprint(eval(expr, globals(), loc));
        else:
            print((eval(expr, globals(), loc))); 
    finally:
        del frame
    
#    locals(), ['opts', 'res'])

def importVarsFromDict(srcDict, destDict):
    for (k,v) in srcDict.items():
        assert k not in destDict;
        destDict[k] = v;
    pass

def tic():
    """
    equivalent to Matlab's tic. It start measuring time.
    returns handle of the time start point.
    """
    global gStartTime
    gStartTime = datetime.utcnow();
    return gStartTime

def toc(prev=None):
    """
    get a timestamp in seconds. Time interval is from previous call of tic() to current call of toc().
    You can optionally specify the handle of the time ending point.
    """
    if prev==None: prev = gStartTime;
    return (datetime.utcnow() - prev).total_seconds();

def printExpr(expr, bPretty=True):
    """ Print the local variables in the caller's frame."""
    from pprint import pprint
    import inspect
    frame = inspect.currentframe()
    try:
        loc = frame.f_back.f_locals
        print(expr, '= ', end=' ') 
        if (bPretty):
            pprint(eval(expr, globals(), loc));
        else:
            print((eval(expr, globals(), loc))); 
    finally:
        del frame


def DictInvert(aDict):
    """\\
    Inverts a dictionary; {(key,value)} is turned into {(value, [key1, key2, ...])}.
    For example, {(1,'a'), (2,'b'), (3,'a')} is turned into {('a',[1,3]), ('b',2)}
    """
    invDict = {};
    for k,v in aDict.items():
        invDict[v] = invDict.get(v, []);
        invDict[v].append(k);

    return invDict;

################################################################################
# numpy 
################################################################################

def calcListStat(aList):
    res = Struct()
    res.nItems = len(aList);
    res.minVal = min(aList);
    res.maxVal = max(aList);
    res.mean = np.mean(aList);
    res.median = np.median(aList);
    sortedList = sorted(aList)

    idx = max([int(round(float(len(aList)) * 0.95)) - 1, 0]);
    res.upper5Perc = sortedList[idx];
    idx = max([int(round(float(len(aList)) * 0.05)) - 1, 0]);
    res.lower5Perc = sortedList[idx];
    return res;

def listFindAll(searchList, elem):
    return [[i for i, x in enumerate(searchList) if x == e] for e in elem]

def fillNFromKFairly(N,K,randStream=ra):
    """
    In [28]: res = fillNFromKFairly(10,3)
    In [29]: res
    Out[29]: array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    """
    perm = ra.permutation(K);
    res = np.zeros(N);
    res = np.remainder(np.arange(N), K);
    res = perm[res];
    return res;

def ListOf2dArrayTo3d(mat):
    tmp = np.zeros((len(mat), mat[0].shape[0], mat[0].shape[1]));
    for i in range(len(mat)): tmp[i,:,:] = mat[i]
    return tmp;

def argkmax(npary,k):
    return npary.argsort()[-k:][::-1]

def get_time_now_kwang():
    return time.strftime('%Y%m%d%a-%H%M%S');

def mahalanobis_norm_sq(x, M):
    assert(x.ndim == 1);
    return np.dot(x, np.dot(M,x));

def mahalanobis_norm_sq_batch_old(X, M):
    """ X is N by d, M is d by d """
    return (np.dot(X, M) * X).sum(1)

def mahalanobis_norm_sq_batch(X, M):
    """ X is N by d, M is d by d """
    return np.einsum('...i,...i', np.dot(X, M), X);

def qoful_construct_query(q_vector, q_mat):
    assert(q_vector.ndim == 1);
    d = len(q_vector);
    q = np.zeros( (d + d**2,1) );
    q[:d,0] = q_vector;
    q[d:,0] = q_mat.ravel();
    return q;

def chooseTopIdxList(ary, topRatio):
    """
      choose topRatio largest member (returns their indexes)
    """
    sidx = np.argsort(ary);
    topIdx = int(np.round(len(ary)*(1-topRatio)));
    return sidx[topIdx:]


def chooseInitPoint(expectedRewardAry, rand_stream=ra):
    """
      choose the initial point by (1) draw rewards (2) choose one randomly
    """
    n = len(expectedRewardAry);
    rewardAry = rand_stream.rand(n) < expectedRewardAry;
    if np.sum(rewardAry) == 0:
      idxAry = np.arange(n);  # when no positive, just choose anything.
    else:
      idxAry = np.where(rewardAry)[0];
    return rand_stream.choice(idxAry);

def chooseInitPoint_v2(expectedRewardAry, nTry=10, rand_stream=ra):
    """
      choose the initial point by (1) draw rewards 10 times (2) choose one randomly
    """
    n = len(expectedRewardAry);
    sumRewardAry = np.zeros(n);
    for iTry in range(nTry):
        rewardAry = rand_stream.rand(n) < expectedRewardAry;
        sumRewardAry += rewardAry;

    if np.sum(rewardAry) == 0:
        idxAry = np.arange(n);  # when no positive, just choose anything.
    else:
        idxAry = np.where(np.max(sumRewardAry) == sumRewardAry)[0]
    return rand_stream.choice(idxAry);

def calcRmse(yhat, y):
    assert yhat.shape == y.shape;
    return np.sqrt(np.mean((yhat - y)**2));

def cmapGen(cmapName='copper', minVal=0.0, maxVal=1.0):
    import matplotlib.pyplot as plt;
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    myCmap = cm = plt.get_cmap(cmapName);
    cNorm  = colors.Normalize(vmin=minVal, vmax=maxVal)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=myCmap);
    return scalarMap;

def cmapGetColor(scalarMap, v):
    return scalarMap.to_rgba(v);

def range_ensure_endpoint(a, b, step):
    """
      In [15]: range_ensure_endpoint(0,10,2)
      Out[15]: [0, 2, 4, 6, 8, 9]
      
      In [16]: range_ensure_endpoint(0,9,2)
      Out[16]: [0, 2, 4, 6, 8]
      
      In [18]: range(0,9,2)
      Out[18]: [0, 2, 4, 6, 8]
      
      In [19]: range(0,10,2)
      Out[19]: [0, 2, 4, 6, 8]
    """
    aList = list(range(a,b,step));
    if (aList[-1] != b-1):
        aList += [b-1];
    return aList;

def translateIndex(ia, ib):
    """ #- ia and ib can be string lists.
        ia = np.array([2,4,5,1,0,3]);
        ib = np.array([5,3,4,2,0,1]);
        b2a,a2b = translateIndex(ia,ib);
        assert np.all(ia == ib[b2a])
    """
    sa = np.argsort(ia);
    sb = np.argsort(ib);
    inv_sa = np.zeros_like( sa );
    inv_sa[sa] = np.arange(len(ia));

    b2a = sb[inv_sa];
    a2b = np.zeros_like( sa );

    a2b[b2a] = np.arange(len(ia));
    return b2a, a2b

def gen_seeds(seed, n_seeds):
    ra.seed(seed);
    return ra.randint(0,np.iinfo(np.uint32).max+1, n_seeds);

def kjunSeed(baseSeed, i):
    """
        generates i-th seeds
        Due to the way it generates the seed, do not use i that is too large..
    """
    assert i <= 100000
    rs = ra.RandomState(baseSeed);
    randVals = rs.randint(np.iinfo(np.uint32).max+1, size=i+1);
    return randVals[-1];

def kjunSeedList(baseSeed, n):
    """
        generates n seeds
        Due to the way it generates the seed, do not use i that is too large..
    """
    assert n <= 100000
    rs = ra.RandomState(baseSeed);
    randVals = rs.randint(np.iinfo(np.uint32).max+1, size=n);
    return randVals;

def nans(*args):
    ary = np.zeros(*args);
    ary.fill(np.nan);
    return ary;

def dstack_product(x, y):
    return np.dstack(numpy.meshgrid(x, y)).reshape(-1, 2)



################################################################################
# Quadratic optimization
################################################################################

def projectOntoL1Ball(v, b):
    """
    PROJECTONTOL1BALL Projects point onto L1 ball of specified radius.

    w = ProjectOntoL1Ball(v, b) returns the vector w which is the solution
    to the following constrained minimization problem:

     min   ||w - v||_2
     s.t.  ||w||_1 <= b.

    That is, performs Euclidean projection of v to the 1-norm ball of radius
    b.

    Author: John Duchi (jduchi@cs.berkeley.edu)
    """
    assert b >= 0, 'Radius of L1 ball is negative';
    assert v.dtype == np.float64;
    if (np.linalg.norm(v,1) < b):
        return v;
    u = np.abs(v);    # this makes a copy
    u[::-1].sort();   # reverse sorting (in-place sort)
    sv = np.cumsum(u);
    rho = np.where(u > ((sv-b) / np.arange(1,len(u)+1,dtype=np.float64)))[0][-1];
    th = np.maximum(0.0, (sv[rho] - b) / (rho+1));
    w = np.sign(v) * np.maximum(abs(v) - th, 0.0);
    return w;

class QuadOptimData(object):
    def __init__(self, dim):
        self.dim = dim;
        self.A = np.zeros((dim,dim));
        self.b = np.zeros(dim);

    def set_A(self, A):
        self.A[:,:] = A;

    def set_b(self, b):
        self.b[:] = b;

def objQuad(th, data, opt):
    f = None;
    g = None;
    v = dot(data.A,th);
    if (opt == 1 or opt == 3):
        f = .5*dot(th,v) + dot(th,data.b); #data.x) - dot(data.thHat,v);
    if (opt == 2 or opt == 3):
        g = v + data.b;
    return f,g;

def minFuncQuadL1Options():
    ret = dict();
    ret['debug'] = False;
    ret['maxIter'] = 400;
    ret['maxLineSearch'] = 10;
    ret['tolX'] = 1e-7;
    ret['tolObj'] = 1e-7;
    ret['alpha0'] = .01; # 2.0/maxEigVal(data.A) is a good heuristic.
    ret['line_c'] = 1e-3; # line_* is for line search
    ret['line_tau'] = .1;
    ret['line_tau0'] = 1.5; # factor to multiply before line search 

    return ret;

def minFuncQuadL1(qoData, maxL1Norm, th0, opt):
    myObj = lambda th, opt: objQuad(th, qoData, opt); 
    
    debug = opt['debug'];
    th = th0;
    if (debug):
        nLineSearchAry = [];
        objValAry = [];
        alphaAry = [];
    maxLineSearch = opt['maxLineSearch'];
    tolX = opt['tolX'];
    tolObj = opt['tolObj'];
    alphaOld = opt['alpha0'];
    line_c = opt['line_c'];
    line_tau = opt['line_tau'];
    line_tau0 = opt['line_tau0'];
    fCnt = 0;
    gCnt = 0;
    bConvergedFVal = False;
    bConvergedX = False;
    #tic();
    f, trash = myObj(th,1);
    for k in range(opt['maxIter']):
        trash, g = myObj(th,3); 
        p = -g;    
#        p = -np.dot(qoData.invA, g);    
        gCnt += 1;

        #- perform backtracking
        alpha = line_tau0 * alphaOld;
        thCur = th;
        for cnt in range(1,maxLineSearch+1):
#         cnt = 1;
#         while cnt <= maxLineSearch:
            thNew = projectOntoL1Ball(thCur + alpha*p, maxL1Norm);
            fNew, trash = myObj(thNew, 1);
            fCnt += 1;
            if (fNew < f - alpha*line_c):
                break;
            elif la.norm(thNew - thCur) / (1+la.norm(thCur)) < tolX:
                if (fNew > f):
                    thNew = thCur;
                    fNew = f;
                break;
            alpha *= line_tau;
#            cnt += 1;

        if (debug):
            nLineSearchAry.append(cnt);
            objValAry.append(f);
            alphaAry.append(alpha);

        if (abs(f - fNew) / (1+abs(f)) <= tolObj):
            bConvergedFVal = True;
            break;
        elif la.norm(thNew - th) / (1 + la.norm(th)) < tolX: 
            bConvergedX = True;
            break;

        th = thNew;
        f = fNew;
        alphaOld = alpha;
    #- thNew: the solution

    #printExpr('toc()');
    iterCnt = k + 1;

    info = {'iterCnt':iterCnt, 'fCnt':fCnt, 'gCnt':gCnt, 'bConverged': bConvergedFVal or bConvergedX};
    
    debugDict = None;
    if (debug):
        info['debugDict'] = {'nLineSearchAry':nLineSearchAry, 'objValAry':objValAry, 'alphaAry':alphaAry }; 
        return thNew, fNew, info;
    else:
        return thNew, fNew, info;



################################################################################
#- for error bound / ml evaluation
################################################################################

def confidenceFactor(N, alpha=0.05):
    return st.t.isf(alpha/2.0, N-1)

def getDeviation(ary, alpha=0.05):
    N = len(ary);
    return confidenceFactor(N, alpha=alpha) * (ary.std(ddof=1) / np.sqrt(N));

def getDeviationMat(mat, alpha=0.05):
    """
    ary: (nMethod) by (nTry)
    """
    nMethod, nTry = mat.shape;
    mat = np.array(mat);
    std = mat.std(1, ddof=1);
    return confidenceFactor(nTry, alpha=alpha) * std / np.sqrt(nTry);

def getErrorBar(ary, alpha=0.05):
    ary = np.array(ary);
    me = ary.mean();
    st = ary.std(ddof=1);
    dev = st / np.sqrt(len(ary)) * confidenceFactor(len(ary), alpha);
    return me, dev

def getErrorBarMat(mat, alpha=0.05):
    me = np.mean(np.array(mat),1);
    return me, getDeviationMat(mat, alpha); 

def evalTestSignificance(aAry, bAry, alpha=0.05):
    diffAry = aAry - bAry;

    n = len(aAry);
    threshold = st.t.isf(alpha/2.0, n-1);
    me = diffAry.mean();
    std = diffAry.std(ddof=1);
    x = diffAry - me;

    dev = np.abs(me / (std / np.sqrt(n)));
    if (dev == np.nan):
        tf = false;
    else:
        tf = (dev > threshold);

    return tf, dev, threshold;

def evalTestSignificanceMat(scoreMat, alpha=0.05, maxOrMin='max'):
    assert maxOrMin in ['min', 'max'];
    nMethod,nTry = scoreMat.shape;

    me = np.mean(scoreMat,1);
    # st = np.std(scoreMat,1);

    significanceMat = np.full( (nMethod,nMethod), False, dtype=bool);
    # devAry = evalCalcDeviation(scoreMat, alpha=alpha)';

    for i in range(nMethod-1):
        for j in range(i+1,nMethod):
            tf, dev, thres = evalTestSignificance(scoreMat[i,:], scoreMat[j,:], alpha=alpha);
            significanceMat[i,j] = tf;
            significanceMat[j,i] = tf;

    if (maxOrMin == 'max'):
        bestIdx = me.argmax()
    else:
        bestIdx = me.argmin()
    
    indistinctBestAry = np.where(significanceMat[bestIdx,:] == False)[0];

    resDict = {};
    resDict['mean'] = me;
    resDict['alpha'] = alpha;
    resDict['significanceMat'] = significanceMat;
    return bestIdx, indistinctBestAry, resDict;


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = la.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def indicator(i,d):
    tmp = np.zeros(d, dtype=float)
    tmp[i] = 1.0
    return tmp



#------------- stdout redirector
from contextlib import contextmanager

@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
