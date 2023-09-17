#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import scipy as sp
import scipy.stats as st
import itertools as it

# Nemenyi post-hoc 
from statsmodels.stats.libqsturng import qsturng, psturng


def nym_permit(ranks,control=None):
        
    k = len(ranks)

    values = list(ranks.values())
    keys = list(ranks.keys())

    if control is not None:
        control_i = keys.index(control)
        comparisons = [keys[control_i] + " // " + keys[i] for i in range(k) if i != control_i]
        z_values = [abs(values[control_i] - values[i]) for i in range(k) if i != control_i]
    else:
        versus = list(it.combinations(range(k), 2))
        comparisons = [keys[vs[0]] + " // " + keys[vs[1]] for vs in versus]
        z_values = [abs(values[vs[0]] - values[vs[1]]) for vs in versus]
 
    p_values = [2*(1-st.norm.cdf(abs(z))) for z in z_values]

# Sort by p_value so that p_0 < p_1, and make_df
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))

    return mknymdf(comparisons, p_values, nymt=z_values)



# make dataframes to return from post-hocs
def mknymdf(avsb, p_vals, nymt):

# k=len(ranks)
# split back to list from rankings
    qq=[]
    for e in range(len(avsb)):
        tmp=avsb[e].split()
        qq.append(tmp[0])
        qq.append(tmp[2])
    ranks = list(set(qq))
    k=len(ranks)

# properly, m = int(k*(k-1)/2.) where k=len(ranks) == len(p_values)
#           len(p_vals) also works for control group tests (k-1)

    n = len(p_vals)

    print("Classifiers:",k,"   Tests:",n)

    bdun_adj = [min((n)*p_value, 1) 
                for p_value in p_vals]

    sidk_adj = [min(1-(1-p_value)**(n), 1) 
                for p_value in p_vals]

## -- nemenyi_test -- ## 
#   t-values are compared with the significance level: 
#            AGREES WITH CD METHOD 
#   psturng() return values are 'array-like' 
#             {strange: some float, some [float]}

    t_values = [psturng((nymt[z] * np.sqrt(2.)), k, np.inf) 
                for z in range(len(nymt))]
# normalise
    for p in range(len(t_values)):
        if isinstance(t_values[p], np.ndarray):
            t_values[p] = t_values[p][0]

    ret_df = pd.DataFrame({"lookup": nymt,
                           "p_noadj": p_vals,
                           "ap_Nymi": t_values, 
                           "ap_BDun": bdun_adj,
                           "ap_Sdak": sidk_adj},
                          index=avsb)
    return ret_df



def nym_psig(indf, alpha=0.05):
    
    pvals_df = indf.loc[indf['p_noadj'] < alpha]
# avsb
    az = pvals_df.index.values.tolist()
# pvals
    psg = list(pvals_df['p_noadj'])
# zvals
    zsg = list(pvals_df['lookup'])

# not sig - for analysis
    rz = indf.index.values.tolist()
    sx = [x for x in rz if x not in az]
    print("Significant:",len(az),"   Not:",len(sx))    

    return mknymdf(az, psg, nymt=zsg)
