from myutils3_v2 import *
from collections import OrderedDict
from blbandits3 import *
from collections import namedtuple

ParamBlTwoStage_ = namedtuple("ParamBlTwoStage_", "C_T1, multiplier")
class ParamBlTwoStage(ParamBlTwoStage_):
    def shortstr(self):
        return "C=%.2g,m=%.2g" % (self.C_T1, self.multiplier)

ParamBlOneStage_ = namedtuple("ParamBlOneStage_", "multiplier")
class ParamBlOneStage(ParamBlOneStage_):
    def shortstr(self):
        return "m=%.2g" % (self.multiplier)

ParamBlOful_ = namedtuple("ParamBlOful_", "multiplier")
class ParamBlOful(ParamBlOful_):
    def shortstr(self):
        return "m=%.2g" % (self.multiplier)

ParamBlOns_ = namedtuple("ParamBlOns_", "multiplier")
class ParamBlOns(ParamBlOful_):
    def shortstr(self):
        return "m=%.2g" % (self.multiplier)


def paramGetList(tuningGridIdx, algoName):
    dd = [paramGetList0, paramGetList1]
    return dd[tuningGridIdx](algoName)

def paramGetList0(algoName):
    paramGrid = OrderedDict()
# 	baseLambdaGrid = 10.0**np.array([-2,-1,0,1,2])
# 	baseLambdaGridExtended = 10.0**np.array([-4,-3,-2,-1,0,1,2])
    baseMultiplierGrid = 10.0**np.array([-np.inf, -4,-3,-2,-1,0])
    baseCT1Grid = 10.0**np.array([.0,.5,1.,1.5])

    # print("DEBUG")
    # print("DEBUG")
    # print("DEBUG")
    # baseMultiplierGrid = 10.0**np.array([-np.inf,1])
    # baseCT1Grid = 10.0**np.array([-0.5,0])

    if   algoName == "bloful":
        paramGrid['multiplier'] = baseMultiplierGrid
        paramclass = ParamBlOful
    elif algoName.startswith("bltwostage"):
        paramGrid['C_T1'] = baseCT1Grid
        paramGrid['multiplier'] = baseMultiplierGrid
        paramclass = ParamBlTwoStage
        pass
    elif algoName.startswith("blonestage"):
        paramGrid['multiplier'] = baseMultiplierGrid
        paramclass = ParamBlOneStage
    elif algoName.startswith("blons"):
        paramGrid['multiplier'] = baseMultiplierGrid
        paramclass = ParamBlOns
    else:
        raise ValueError()

    [keys,perms] = paramGetPermutations(paramGrid)
    return [paramclass(*x) for x in perms]

def paramGetList1(algoName):
    """ a more fine-grained """
    paramGrid = OrderedDict()
# 	baseLambdaGrid = 10.0**np.array([-2,-1,0,1,2])
# 	baseLambdaGridExtended = 10.0**np.array([-4,-3,-2,-1,0,1,2])
    baseMultiplierGrid = 10.0**np.array([-2,-1.5,-1,-.5,0])
    baseCT1Grid = 10.0**np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # print("DEBUG")
    # print("DEBUG")
    # print("DEBUG")
    # baseMultiplierGrid = 10.0**np.array([-np.inf,1])
    # baseCT1Grid = 10.0**np.array([-0.5,0])

    if   algoName == "bloful":
        paramGrid['multiplier'] = baseMultiplierGrid
        paramclass = ParamBlOful
    elif algoName.startswith("bltwostage"):
        paramGrid['C_T1'] = baseCT1Grid
        paramGrid['multiplier'] = baseMultiplierGrid
        paramclass = ParamBlTwoStage
    elif algoName.startswith("blonestage"):
        paramGrid['multiplier'] = baseMultiplierGrid
        paramclass = ParamBlOneStage
        pass
    elif algoName.startswith("blons"):
        paramGrid['multiplier'] = baseMultiplierGrid
        paramclass = ParamBlOns
    else:
        raise ValueError()

    [keys,perms] = paramGetPermutations(paramGrid)
    return [paramclass(*x) for x in perms]

# def paramGetGrid(algoName):
# 	paramGrid = OrderedDict()
# # 	baseLambdaGrid = 10.0**np.array([-2,-1,0,1,2])
# # 	baseLambdaGridExtended = 10.0**np.array([-4,-3,-2,-1,0,1,2])
# 	baseMultiplierGrid = 10.0**np.array([-np.inf, -4,-3,-2,-1,0,1])
# 	baseCT1Grid = 10.0**np.array([-1.0,-0.5,0,0.5,1.0])
# 
# 	# print("DEBUG")
# 	# print("DEBUG")
# 	# print("DEBUG")
# 	# baseMultiplierGrid = 10.0**np.array([-np.inf,1])
# 	# baseCT1Grid = 10.0**np.array([-0.5,0])
# 
# 	if   algoName == "bloful":
# 		paramGrid['multiplier'] = baseMultiplierGrid
# 	elif algoName.startswith("bltwostage"):
# 		paramGrid['C_T1'] = baseCT1Grid
# 		paramGrid['multiplier'] = baseMultiplierGrid
# 		pass
# 	else:
# 		raise ValueError()
# 
# 	return paramGrid;

def paramGetPermutations(paramGrid):
    #- DO RECURSION!!
    keys = paramGrid.keys()
    assert type(paramGrid) is OrderedDict
    valsList = [paramGrid[k] for k in keys];
    perms = getPermutations(0, valsList);

    return keys, perms

def paramGetNamespaceList(keys, perms):
    """ we get a list of namespace{ param1name: v1, param2name: v2, ... }
    """
    keys = list(keys)
    retList = []
    for i in range(len(perms)):
        sns = SimpleNamespace()
        for j in range(len(keys)):
            sns.__dict__[keys[j]] = perms[i][j]
        retList.append( sns )
    return retList

def getPermutations(level, valsList):
    """
    In : ff(0,[[1,2],[3,4]])
    Out: [[1, 3], [1, 4], [2, 3], [2, 4]]
    """
    # ipdb.set_trace();
    if (level >= len(valsList)):
        return [];
    
    aList = [];
    suffixList = getPermutations(level+1, valsList);
    for v in valsList[level]:
        if (len(suffixList) == 0):
            aList.append( [v] );
        else:
            for suffix in suffixList:
                aList.append( [v] + suffix );
    return aList;

def dataFactory(dataname, dataopts):
    """
    """
    o = dataopts
    if   dataname == 'sphericalgaussian':
        data = SphericalGaussian(o.R, o.r)
        data.gen_data(o.d1, o.d2, o.N1, o.N2, o.S_2norm, armtype=o.armtype)
    elif dataname == 'movielens':
        data = MovieLense('../../data/movielens/out/movielens_128_mc.pkl',o.R)
        data.gen_features()
    else:
        raise ValueError()
    return data

def banditFactory(data, algoName, algoParam, exprOpts):
    lam = exprOpts.lam
    if algoName == "bloful":
        Sp = np.sqrt(lam)*data.S_F
        algo = BilinearOful(data.X, data.Z, lam, data.R, Sp, 
                            multiplier=algoParam.multiplier)
    elif algoName == "bltwostage":
        svals = sla.svdvals(data.Th)[:data.r]
        algo = BilinearTwoStage(data.X, data.Z, 
                                lam, data.R, 
                                data.S_F,
                                svals.max(), svals.min(), data.r, 
                                algoParam.C_T1, exprOpts.T,
                                multiplier=algoParam.multiplier,
                                SpType=None)
    elif algoName.startswith("bltwostage"):
        tokens = algoName.split('-')
        if len(tokens) == 2:
            algoMatrixCompletion = 'optspace'
            SpType = tokens[1][3:]
        elif len(tokens) == 3:
            algoMatrixCompletion = tokens[1]
            SpType = tokens[2][3:]
            pass
        else:
            raise ValueError()
        svals = sla.svdvals(data.Th)[:data.r]
        algo = BilinearTwoStage(data.X, data.Z, 
                                lam, data.R, 
                                data.S_F,
                                svals.max(), svals.min(), data.r, 
                                algoParam.C_T1, exprOpts.T,
                                multiplier=algoParam.multiplier,
                                SpType=SpType,
                                algoMatrixCompletion=algoMatrixCompletion)
    # elif algoName == "bltwostage-sp_simple":
    #     svals = sla.svdvals(data.Th)[:data.r]
    #     algo = BilinearTwoStage(data.X, data.Z, 
    #                             lam, data.R, 
    #                             data.S_F,
    #                             svals.max(), svals.min(), data.r, 
    #                             algoParam.C_T1, exprOpts.T,
    #                             multiplier=algoParam.multiplier,
    #                             SpType='simple')
    # elif algoName == "bltwostage-sp_simple2":
    #     svals = sla.svdvals(data.Th)[:data.r]
    #     algo = BilinearTwoStage(data.X, data.Z, 
    #                             lam, data.R, 
    #                             data.S_F,
    #                             svals.max(), svals.min(), data.r, 
    #                             algoParam.C_T1, exprOpts.T,
    #                             multiplier=algoParam.multiplier,
    #                             SpType='simple2')
    # elif algoName == "bltwostage-sp_simple3":
    #     svals = sla.svdvals(data.Th)[:data.r]
    #     algo = BilinearTwoStage(data.X, data.Z, 
    #                             lam, data.R, 
    #                             data.S_F,
    #                             svals.max(), svals.min(), data.r, 
    #                             algoParam.C_T1, exprOpts.T,
    #                             multiplier=algoParam.multiplier,
    #                             SpType='simple3')
    elif algoName == "blonestage-sp_simple2":
        svals = sla.svdvals(data.Th)[:data.r]
        algo = BilinearOneStage(data.X, data.Z, 
                                lam, data.R, 
                                data.S_F,
                                svals.max(), svals.min(), data.r, 
                                exprOpts.T,
                                multiplier=algoParam.multiplier,
                                SpType='simple2')
    elif algoName == "blonestage-sp_simple3":
        svals = sla.svdvals(data.Th)[:data.r]
        algo = BilinearOneStage(data.X, data.Z, 
                                lam, data.R, 
                                data.S_F,
                                svals.max(), svals.min(), data.r, 
                                exprOpts.T,
                                multiplier=algoParam.multiplier,
                                SpType='simple3')
    elif algoName == "blons":
        S_star = la.norm(data.Th, "nuc")
        algo = BilinearGlocNuclear(data.X, data.Z, lam, data.R, 
                                   S_star, 
                                   multiplier=algoParam.multiplier, 
                                   calc_radius_version=3)
    elif algoName == "blons-naive":
        S_star = la.norm(data.Th, "nuc")
        algo = BilinearGlocNuclear(data.X, data.Z, lam, data.R, 
                                   S_star, 
                                   flags={'bNaive':True},
                                   multiplier=algoParam.multiplier, 
                                   calc_radius_version=3)
    else:
        raise ValueError();
    return algo;


def reduceOutputSize(out):
    # out.opts
    # out.res
    # out.res.arms  (tryIdx, paramIdx, t)
    # out.res.times
    # out.res.expected_rewards (tryIdx, N1 by N2)

    res = out.res
    T = out.opts.T

    tAry = np.concatenate( (np.arange(1,T+1,int(np.sqrt(T))), [T]) );
    tAry = np.unique(tAry)

    # first, compute the cumulative rewards.
    cum_expected_rewards = [] # cumulative expected rewards
    best_expected_rewards = [] # best expected rewards
    cum_regrets = []
    for tryIdx in range(len(res.arms)):
        er = res.expected_rewards[tryIdx] # N1 by N2

        #- find the best expected rewards
        my_ber = np.max(er)
        best_expected_rewards.append( my_ber )

        #- prepare
        mat = np.array([ [ er[pair] for pair in aList ] for aList in res.arms[tryIdx]])

        #- cumulative expected rewards
        my_cer = mat.cumsum(1)
        cum_expected_rewards.append( my_cer )

        #- cumulative regret
        my_cum_regret = (my_ber - mat).cumsum(1)
        cum_regrets.append( my_cum_regret )

    newres = SimpleNamespace()
    from copy import deepcopy
    newres = deepcopy(res)
    newres.tAry = tAry
    newres.arms = []
    newres.best_expected_rewards = best_expected_rewards
    newres.cum_expected_rewards = [] #cum_expected_rewards
    newres.cum_regrets = [] #cum_regrets

    for tryIdx in range(len(res.arms)):
        tmp = [ [aList[t-1] for t in tAry] for aList in res.arms[tryIdx]]
        newres.arms.append( tmp )

        tmp = cum_expected_rewards[tryIdx][:,tAry-1]
        newres.cum_expected_rewards.append( tmp )
        
        tmp = cum_regrets[tryIdx][:,tAry-1]
        newres.cum_regrets.append( tmp )

    newout = deepcopy(out)
    newout.res = newres
    newout.opts = out.opts

    return newout
