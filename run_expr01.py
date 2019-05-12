"""
first experiment for oful vs 2stage
"""
from myutils3_v2 import *
from blbandits3 import *
import bleval
from expr01_defs import *
from types import SimpleNamespace
from tqdm import tqdm
import sys
np.set_printoptions(precision=4, threshold=200)

# A="bloful";                ipy3 run_expr01.py ${A} -- -l 0.1 -p lam ; ipy3 run_expr01.py ${A} -- -r 4 -p r
# A="bltwostage";            ipy3 run_expr01.py ${A} -- -l 0.1 -p lam ; ipy3 run_expr01.py ${A} -- -r 4 -p r
# A="bltwostage-sp_simple";  ipy3 run_expr01.py ${A} -- -l 0.1 -p lam ; ipy3 run_expr01.py ${A} -- -r 4 -p r
# A="bltwostage-sp_svalmax"; ipy3 run_expr01.py ${A} -- -l 0.1 -p lam ; ipy3 run_expr01.py ${A} -- -r 4 -p r

# A="bloful";                ipy3  run_expr01.py ${A} -- -a rademacher ; A="bltwostage";            ipy3  run_expr01.py ${A} -- -a rademacher ; A="bltwostage-sp_simple";  ipy3  run_expr01.py ${A} -- -a rademacher ; A="bltwostage-sp_svalmax"; ipy3  run_expr01.py ${A} -- -a rademacher 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("algo") 
parser.add_argument("-l", "--lam", nargs='?', default=None)
parser.add_argument("-r", nargs='?', default=None)
parser.add_argument("-R", nargs='?', default=None)
parser.add_argument("-d", nargs='?', default=None)
parser.add_argument("-p", "--prefix", nargs='?', default=None)
parser.add_argument("-a", "--armtype", nargs='?', default=None)
parser.add_argument("-dt", "--dataname", nargs="?", default='sphericalgaussian')
parser.add_argument("-T", nargs='?', default=None)
parser.add_argument("-tg", "--tuning_grid", nargs='?', default=None) # tuninggrid
parser.add_argument("-n", "--nTry", nargs='?', default="30")
args = parser.parse_args()

#- first, make some examples
algoName = args.algo
fNamePrefix = args.prefix
armtype = args.armtype
armtype = "gaussian" if armtype is None else armtype
assert armtype in [None, "gaussian", "rademacher", "rademacher2"]

opts = SimpleNamespace()
opts.nTry = int(args.nTry)
opts.gSeed = 119
opts.dataSeed = 99
opts.R = float(args.R) if (args.R is not None) else 0.05

#--- options for toy
opts.dataopts = SimpleNamespace()
if   args.dataname == 'sphericalgaussian':
    d = int(args.d) if args.d is not None else 16
    ratio_N_to_d = 2
    opts.dataopts.d1 = d
    opts.dataopts.d2 = d
    opts.dataopts.r = int(args.r) if(args.r is not None) else 2
    opts.dataopts.N1 = ratio_N_to_d*d
    opts.dataopts.N2 = ratio_N_to_d*d
    opts.dataopts.S_2norm = 1.0
    opts.dataopts.armtype = armtype
    opts.dataopts.R = opts.R
elif args.dataname == 'movielens':
    opts.dataopts.R = opts.R
else:
    raise ValueError()

opts.T = int(args.T) if (args.T is not None) else 10000
opts.lam = float(args.lam) if (args.lam is not None) else 1.0
opts.args = args
opts.tuningGrid = int(args.tuning_grid) if (args.tuning_grid is not None) else 0

# # For debugging
# print("DEBUG"); print("DEBUG"); print("DEBUG")
# opts.nTry = 2

printExpr('opts')

resList = []
res = SimpleNamespace()
res.arms = []
res.times = []
res.expected_rewards = []
res.dbgDicts = []
for tryIdx in tqdm(range(opts.nTry)):
    print('')
    print('#'*80)
    print('#----- tryIdx = %5d' % tryIdx)

    #- data
    ra.seed(kjunSeed(opts.dataSeed, tryIdx))
    data = dataFactory(args.dataname, opts.dataopts)
    printExpr("(la.norm(data.Th), la.norm(data.Th,2))")
    printExpr('data.expt_reward[0,0]')

    #--- get the list of parameters
    paramList = paramGetList(opts.tuningGrid, algoName)
    printExpr('paramList')

    ra.seed(kjunSeed(opts.gSeed,tryIdx))

    #--- for each parameter tuple
#    armMat = nans( (len(paramNamespaceList), T) )
    armMat = [None]*len(paramList)
    dbgDictAry = []
    timeAry = []
    for paramIdx, algoParam in enumerate(paramList):
        print('\n#- paramIdx = %5d' % paramIdx)
        printExpr('algoParam')
        algo = banditFactory(data, algoName, algoParam, opts)

        tt = tic()
        [rewardAry, armPairAry, dbgDict] = run_bilinear_bandit(algo, data, opts.T)
        elapsed = toc(tt)

        cumExpectedRewards = np.sum([data.expt_reward[row[0],row[1]] for row in armPairAry])
        cumExpectedRegret = data.get_best_reward() * opts.T  - cumExpectedRewards
        printExpr('elapsed')
        printExpr('cumExpectedRewards')
        printExpr('cumExpectedRegret')

        armMat[paramIdx] = [(row[0],row[1]) for row in armPairAry]
        dbgDictAry.append( dbgDict )
        timeAry.append( elapsed )
        sys.stdout.flush() # to work better with tee

    res.arms.append( armMat )
    res.times.append( timeAry )
    res.expected_rewards.append( data.expt_reward )
    res.dbgDicts.append( dbgDictAry )

output = SimpleNamespace()
output.res = res
output.opts = opts
output.paramList = paramList

newout = reduceOutputSize(output)
if (fNamePrefix is None):
    SavePickle('%s-%s.pkl' % (get_time_now_kwang(), algoName), newout)
else:
    SavePickle('%s-%s-%s.pkl' % (fNamePrefix, get_time_now_kwang(), algoName), newout)

# ################################################################################
# #  UC
# ################################################################################
# 
# #- learner 1
# # Sp = np.sqrt(lam)*la.norm(data.Th,'fro')
# # learner = BilinearOful(data.X, data.Z, lam, R, Sp)
# 
# #- in practice, I should set 'sval min' = 'sval max'.
# svals = sla.svdvals(data.Th)[:opts.r]
# learner = BilinearTwoStage(data.X, data.Z, opts['lam'], opts['R'], svals.max(), svals.min(), r, C_T1, T)
# 
# rewardAry, armPairAry, dbgDict = run_bilinear_bandit(learner, data, T)
# 
# cumregret = bleval.calc_regret(data, armPairAry)
# 
# import matplotlib.pyplot as plt
# plt.ion()
# plt.figure()
# plt.plot(cumregret)
# plt.figure()
# plt.loglog(cumregret)
# plt.grid('on')
# 
# t2 = T-1
# t1 = int(T//2-1)
# printExpr('np.log(cumregret[t2]/cumregret[t1])/np.log(t2/t1)')
# 
# 
# 
