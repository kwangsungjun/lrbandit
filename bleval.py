from myutils3_v2 import *

def calc_regret(data_obj, arm_pair_ary):
    do = data_obj;
    best_reward = np.max(do.expt_reward);
    raveled = np.ravel_multi_index(arm_pair_ary.T, do.expt_reward.shape)
    return np.cumsum(best_reward - do.expt_reward.ravel()[raveled]);

