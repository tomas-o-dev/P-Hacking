#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import scipy as sp
import scipy.stats as st
import itertools as it

# Nemenyi post-hoc 
from statsmodels.stats.libqsturng import qsturng, psturng

# Cochran's Q, McNemar post-hoc 
from mlxtend.evaluate import mcnemar_table, mcnemar, cochrans_q
from sklearn.preprocessing import LabelEncoder

####

## run_friedman(indf, alpha=0.05):
#    Wrapper for friedman_test() from stac library
#                https://tec.citius.usc.es/stac/doc/
#
# Parameters
# ----------
# indf : pd.Dataframe
#    The sample measurements for each group
# alpha : float, default = 0.05
#    Significance threshold
#
# Returns
# -------
# sig : boolean
#    Reject H0?   
# rptstr : String
#    Accept or Reject H0 
# rankdic : dict
#    dict of 'pivotal quantities' for the post-hoc tests
# avg_ranks : dict
#    dict of average ranks for analysis after the post-hoc tests


## ph_pvals(ranks,control=None,nmyi=False,shaf=False):
#    Post-Hoc p_values adjusted for multiple testing 
#
# Parameters
# ----------
# ranks : dict
#    A dict with format 'groupname':'pivotal quantity' 
#    returned from Freidman test (rankdic)
# control : string, default = None
#    'groupname' for one-to-all (control group) comparisons
#    default is all-vs-all comparison
# nmyi : Boolean, default = False
#     Run the Nemenyi test
#     Note: nemenyi_test is not appropriate for one.vs.all
# shaf : Boolean, default = False
#     Run the Schaffer_static_test
#     Note: schaffer_static uses a recursive call; this causes
#           internal python multithreading to go wildly oversubscribed
#           when there are more than 18 classfiers to compare
#
# Returns
# ----------
# pd.Dataframe with adjusted p_values from various methods
# -- default (nmyi==shaf==False) --
#    p_noadj : no adjustment for multiple testing
#    ap_BDun : Bonferroni-Dunn (single-step)
#    ap_Sdak : Sidak (single-step)
#    ap_Holm : Holm (step-down)
#    ap_Finr : Finner (step-down)
#    ap_Hoch : Hochberg (step-up)
#    ap_Li   : Li (step-up)
# -- Nemenyi test (nmyi=True) --
#    ap_Nymi : Nemenyi (single-step)
#    ap_BDun : Bonferroni-Dunn
#    ap_Sdak : Sidak 
# -- Schaffer test (shaf=True) --
#    ap_Shaf : Schaffer static (step-down)
#    ap_Holm : Holm 
#    ap_Finr : Finner


## cq_mph(y_test,clf_preds,cq=True,control=None,alpha=0.05):
#    Cochrans_Q omnibus, McNemar post-hoc
#    requires mlxtend.evaluate
#
# Parameters
# ----------
# y_test: array-like, shape=[n_samples]
#    True class labels as 1D NumPy array.
# clf_preds : nested list, len=number of classifiers
#             clf_preds[n][0] : classifier name (string)
#             clf_preds[n][1] : array, shape=[n_samples] 
#                               predicted class labels
# cq : Boolean, default = True
#    Run the Cochrans_Q omnibus test
# control : string, default = None
#    classifier name for one-to-all (control group) 
#                        post-hoc comparisons
#    default is all-vs-all comparison
# alpha : float, default = 0.05
#    Significance threshold
#
# Returns
# ----------
# pd.Dataframe with adjusted p_values for various tests
# -- ph_pvals default --


# filtr_ap2h0(indf, alpha=0.05):
#    Converts dataframe returned by ph_pvals() or cq_mph()
#    from adjusted p_values to T/F for null hypothesis
#
# Parameters
# ----------
# indf : pd.Dataframe
#    Dataframe returned by ph_pvals() or cq_mph()
# alpha : float, default = 0.05
#    Significance threshold
#
# Returns
# -------
# pd.Dataframe with T/F for null hypothesis


# filtr_psig(indf, alpha=0.05):
#    Reduces dataframe returned by ph_pvals() or cq_mph()
#    to have only p_noadj < alpha
#
# Parameters
# ----------
# indf : pd.Dataframe
#    Dataframe returned by ph_pvals() or cq_mph()
# alpha : float, default = 0.05
#    Significance threshold
#
# Returns
# -------
# pd.Dataframe with selected rows


# compare_avgranks(indf, avg_ranks, alpha=0.05):
# takes indf from mkdf plus avg_ranks from freidman test
#
# Parameters
# ----------
# indf : pd.Dataframe
#    Dataframe returned by ph_pvals() or cq_mph()
# avg_ranks : dict
#    dict returned by run_freidman()
# alpha : float, default = 0.05
#    Significance threshold
#
# Returns
# -------
# list sorted by first field


# compare_avgranks_lf(cmp):
# resort list from compare_avgranks()
#
# Parameters
# ----------
# cmp : list
#    list returned by compare_avgranks()
#
# Returns
# -------
# list sorted by last field


# sort_dict_byval(indd, rev=False):
# sort dict by value (dict comprehension)
#
# Parameters
# ----------
# indd : dict
#    intended for string/scalar values ...
# rev : Boolean, default = False
#    true: higest to lowest
#
# Returns
# -------
# dict


####


# Friedman mean ranks test
def friedman_test(*args):

    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v)-1)/2. for v in row])

# used here for chisq
    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]

# used in post-hocs to calculate z_value
    rankings_cmp = [r/np.sqrt(k*(k+1)/(6.*n)) for r in rankings_avg]

    chi2 = ((12*n)/float((k*(k+1))))*((sum(r**2 for r in rankings_avg))-((k*(k+1)**2)/float(4)))

    iman_davenport = ((n-1)*chi2)/float((n*(k-1)-chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k-1, (k-1)*(n-1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def run_friedman(indf, alpha=0.05):
# requires data where the rows are classifiers and the columns are datasets 
    data = np.asarray(indf)
    f, p, ranks, piv = friedman_test(*np.transpose(data))

# post-hocs require a dict of 'pivotal values' from the Friedman test 
    rankdic = {key: piv[i] for i, key in enumerate(list(indf.columns))} 

# analysis can use the dict of average ranks from the Friedman test 
    avg_ranks = {key: ranks[i] for i, key in enumerate(list(indf.columns))} 

    sig = p <= alpha
    rptstr = "Freidman Test\n"
    rptstr += "H0: there is no difference in the means at the "
    rptstr += str((1-alpha)*100) + "% confidence level \n"

    if sig:
        rptstr += "Reject: Ready to continue with post-hoc tests"
    else:
        rptstr += "Accept: No need for post-hoc tests"

    return sig, rptstr, rankdic, avg_ranks


# Post-hoc tests use a dict of 'pivotal quantities' from Freidman test
#     (see above)
#    rankings_cmp = [r/np.sqrt(k*(k+1)/(6.*n)) 
#                    for r in rankings_avg]
#    rankdic = {key: rankings_cmp[i] 
#               for i, key in enumerate(list(indf.columns))} 

def ph_pvals(ranks,control=None,nmyi=False,shaf=False):

    if control is not None and nmyi:
        print("Exception: Nemenyi_test is only appropriate for all.vs.all")
        ret_df = pd.DataFrame({"nym_t": 1, 
                               "rejH0": False},
                              index=['Error'])
        return ret_df
        
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

    if nmyi:
        return mkdf(comparisons, p_values, nymt=z_values)
    else:
        return mkdf(comparisons, p_values, tj=shaf)


# make dataframes to return from post-hocs
def mkdf(avsb, p_vals, nymt=None, tj=False):

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

    holm_adj = [min(max((n-j)*p_vals[j] 
                for j in range(i+1)), 1) 
                for i in range(n)]

    finr_adj = [min(max(1-(1-p_vals[j])**(n/float(j+1)) 
                for j in range(i+1)), 1) 
                for i in range(n)]

    hoch_adj = [min((n-j+1)*p_vals[j-1]
                for j in range(n, i, -1)) 
                for i in range(n)]

    li_adjpv = [p_vals[i]/(p_vals[i]+1-p_vals[-1]) 
                for i in range(n)]

## -- nemenyi_test -- ## 
#   t-values are compared with the significance level: 
#            AGREES WITH CD METHOD 
#   psturng() return values are 'array-like' 
#             {strange: some float, some [float]}

    if nymt is not None:
        t_values = [psturng((nymt[z] * np.sqrt(2.)), k, np.inf) for z in range(len(nymt))]
# normalise
        for p in range(len(t_values)):
            if isinstance(t_values[p], np.ndarray):
                t_values[p] = t_values[p][0]

        ret_df = pd.DataFrame({"p_noadj": p_vals,
                               "ap_Nymi": t_values, 
                               "ap_BDun": bdun_adj,
                               "ap_Sdak": sidk_adj},
                              index=avsb)
        return ret_df

## -- shaffer_static -- ##
## internal python multithreading goes wildly oversubscribed
## when there are more than 18 classfiers to compare

    if tj:
        A = _S(int((1 + np.sqrt(1+4*n*2))/2))  # call recursive 
        t = [max([a for a in A if a <= n-i]) for i in range(n)]

        shaf_adj = [min(max(t[j]*p_vals[j] 
                    for j in range(i+1)), 1) 
                    for i in range(n)]

        ret_df = pd.DataFrame({"p_noadj": p_vals,
                               "ap_Shaf": shaf_adj,
                               "ap_Holm": holm_adj,
                               "ap_Finr": finr_adj},
                              index=avsb)
        return ret_df

##-- general case (e.g., mcnemar) --##
    ret_df = pd.DataFrame({"p_noadj": p_vals,
                           "ap_BDun": bdun_adj,
                           "ap_Sdak": sidk_adj,
                           "ap_Holm": holm_adj,
                           "ap_Finr": finr_adj,
                           "ap_Hoch": hoch_adj,
                           "ap_Li": li_adjpv},
                          index=avsb)
    return ret_df


# recursive helper function for the Shaffer (static) test:
# obtains the number of independent test hypotheses 
# from the number of groups to be compared.
## internal python multithreading goes wildly oversubscribed
## when there are more than 18 classfiers to compare

def _S(k):

    if k == 0 or k == 1:
        return {0}
    else:
        result = set()

## recursive - slow for big jobs but hard to parallelise
## ---------
        for j in reversed(range(1, k+1)):
            tmp = _S(k - j)
            for s in tmp:
                result = result.union({sp.special.binom(j, 2) + s})
## ---------
        return list(result)


# convert p_values to H0 T/F
def filtr_ap2h0(indf, alpha=0.05):
    hodf = indf.copy()
# -- nemenyi_df -- #
    hodf.drop(['lookup'], axis=1, inplace=True, errors='ignore')
# --  -- #
    hodf.columns = hodf.columns.str.replace('ap_', 'H0: ')
    hodf = (hodf>alpha)
    return hodf


def filtr_psig(indf, alpha=0.05):
    if 'p_noadj' not in indf.columns:
        print("Error: Requires dataframe from ph_pvals() or cq_mph()")
        ret_df = pd.DataFrame({"Required": 'p_noadj'},
                              index=['Error'])
        return ret_df    
    
    pvals_df = indf.loc[indf['p_noadj'] < alpha]
# avsb
    az = pvals_df.index.values.tolist()
# pvals
    psg = list(pvals_df['p_noadj'])

# not sig - for analysis
    rz = indf.index.values.tolist()
    sx = [x for x in rz if x not in az]
    print("Significant:",len(az),"   Not:",len(sx))    

    shaf = True if 'ap_Shaf' in pvals_df.columns else False   

    if 'ap_Nymi' in pvals_df.columns:
       print("Exception: Nemenyi_test is only appropriate for all.vs.all")
       print("           Returning standard tests")

    return mkdf(az, psg, tj=shaf)


def compare_avgranks(indf, avg_ranks, alpha=0.05):
    if 'p_noadj' not in indf.columns:
        print("Error: Requires dataframe from ph_pvals()")
        ret_df = pd.DataFrame({"Required": 'p_noadj'},
                              index=['Error'])
        return ret_df    
    
    pz = indf.index.values.tolist()
    cmp = []
    for c in range(len(pz)):
        bb=pz[c].split()
        if avg_ranks[bb[0]] > avg_ranks[bb[2]]:
            rnx=bb[2]+' '+str(avg_ranks[bb[2]])+' // '+str(avg_ranks[bb[0]])+' '+bb[0]
        else:
            rnx=bb[0]+' '+str(avg_ranks[bb[0]])+' // '+str(avg_ranks[bb[2]])+' '+bb[2]
        cmp.append(rnx)
        
    print("Note: differences Down the Columns are NOT significant")
    print("      only differences Across the Rows ARE significant")
    return sorted(cmp)


def compare_avgranks_lf(cmp):
    print("sorted by last field")
    return sorted(cmp, key=lambda t: t.split()[4])


def sort_dict_byval(indd, rev=False):
# sort by value (dict comprehension)
    if rev:
        retd = {x: v for v, x in sorted(((value, key) for (key, value) in indd.items()), reverse=True)}
    else:
        retd = {x: v for v, x in sorted((value, key) for (key, value) in indd.items())}
    return retd


# Cochran's Q omnibus test with McNemar post-hoc 
def cq_mph(y_test,clf_preds,cq=True,control=None,alpha=0.05):

# test for numeric labels
    if not isinstance(y_test[0],(int,np.integer)):
        ynum = LabelEncoder().fit_transform(y_test)
    else:
        ynum = y_test

    rnoms = []
    rvals = []
    for r in range(len(clf_preds)):
        rnoms.append(clf_preds[r][0])

        if not(isinstance(clf_preds[0][1][0],(int,np.integer))): 
            rvals.append(LabelEncoder().fit_transform(clf_preds[r][1]))
        else:
            rvals.append(clf_preds[r][1])

    if control is not None:
        ndx = [i for i in range(len(rnoms)) if rnoms[i] == control]
        if len(ndx) != 1:
            print('Error: Control Name not found',ndx)
            ret_df = pd.DataFrame({"Control": 1,
                                   "rejH0": False},
                                  index=['Error'])
            return ret_df      
        else:
            control_i = ndx[0]

## omnibus test ##           
# unpack the list with *arg
    if cq:
        qval, pv = cochrans_q(ynum,*rvals)
        print('Cochran Q Test:',rnoms)
        print('\tp_value =', round(pv,3), 'and ChiSquare =', round(qval,3),"\n")
        rptstr = "H0: there is no difference in performance at the "
        rptstr += str((1-alpha)*100) + "% confidence level\n\t"

        if pv > alpha:
            rptstr += "Accept - No need for post-hoc tests\n"
            print(rptstr)
            ret_df = pd.DataFrame({"p": round(pv,3), 
                                   "rejH0": False},
                                  index=['CochranQ'])
            return ret_df
        else:
            rptstr += "Reject - Continuing with post-hoc tests\n"
            print(rptstr)

## post-hoc ##           
    if control is not None: 
        comparisons = [rnoms[control_i] + " // " + rnoms[i] for i in range(len(rnoms)) if i != control_i]
        pred_values = [ [ rvals[control_i], rvals[i] ] for i in range(len(rvals)) if i != control_i ]
    else:
        versus = list(it.combinations(range(len(rvals)), 2))
        comparisons = [rnoms[vs[0]] + " // " + rnoms[vs[1]] for vs in versus]
        pred_values = [ [ rvals[vs[0]], rvals[vs[1]] ] for vs in versus ]

    p_values = []
    for r in range(len(pred_values)):
        chisq, pv = mcnemar(mcnemar_table(ynum, *pred_values[r]))
        p_values.append(pv)

# Sort values by p_value so that p_0 < p_1
    p_values, comparisons = map(list, zip(*sorted(zip(p_values, comparisons), key=lambda t: t[0]))) 

    return mkdf(comparisons, p_values)



