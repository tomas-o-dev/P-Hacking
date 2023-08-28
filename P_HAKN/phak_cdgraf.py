#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import math
import itertools as it
import matplotlib.pyplot as plt

from statsmodels.stats.libqsturng import qsturng, psturng


def get_nemenyi_cd(indf, alpha=0.05):
    data = np.asarray(indf)
    n_datasets, n_methods = data.shape
    q_alpha = qsturng(1 - alpha, n_methods, np.inf) / np.sqrt(2)
    critical_difference = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets)) 
    return critical_difference


def nemenyi_cdgraph(indf, alpha=0.05, pngpath=None, imgres=300):
    crit_dis = get_nemenyi_cd(indf)
    rankmat = indf.rank(axis='columns', ascending=False)
# pd.series
    meanranks = rankmat.mean()
    
    # From AutoRank: https://sherbold.github.io/autorank/
    _, names, groups, sigd = get_sorted_rank_groups(crit_dis, meanranks)

    rptstr = "Critical Distance: " + str(round(crit_dis,4)) + "\n"

    if len(groups) == 0:
        rptstr += "\nBased on the Critical Distance, "
        rptstr += "all differences between groups are significant:\n"
    else:
        rptstr += "Significant dfference:\n"
        for s in range(len(sigd)):
            rptstr += "\t" + sigd[s] + "\n"

        rptstr += "The graph shows "
        rptstr += "there are no significant differences\nwithin "
        rptstr += str(len(groups)) + " subgroups of the " + str(len(names)) + " tested:"
   
    cdplot = cd_diagram(crit_dis, meanranks, reverse=True, png=pngpath,  res=imgres)   # , ax=, width=)
    return rptstr, cdplot       # significance, filepath


def get_sorted_rank_groups(cd, meanranks, reverse=False):
    # From AutoRank: https://sherbold.github.io/autorank/
    if reverse:      # for diagram
        sorted_ranks = meanranks.sort_values(ascending=True)
    else:
        sorted_ranks = meanranks.sort_values(ascending=False)
    names = sorted_ranks.index.to_list()        

    groups = []
    sigdif = []
    cur_max_j = -1
    sigg=0
    for i in range(len(sorted_ranks)):
        max_j = None
        for j in range(i + 1, len(sorted_ranks)):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= cd:
                max_j = j
            else:
                sigg += 1
                sgd = names[i] +" // "+ names[j]
                sigdif.append(sgd)    
        if max_j is not None and max_j > cur_max_j:
            cur_max_j = max_j
            groups.append((i, max_j))
    if sigg == 0:
        sgd = "None"
        sigdif.append(sgd)    
    return sorted_ranks, names, groups, sigdif


def cd_diagram(cd, meanranks, reverse=True, ax=None, width=None, png=None, res=300):
    """
    From AutoRank: https://sherbold.github.io/autorank/
    Creates a Critical Distance diagram.
    ##    ax (Axis, default=None):
    ##        Matplotlib axis to which the results are added. 
    ##        A new figure with a single axis is created if None.

    ##    width (float, default=None):
    ##        Specifies the width of the created plot is not None. By default, we use a width of 6. 
    ##        The height is automatically determined, based on the type of plot and the number of 
    ##        populations. This parameter is ignored if ax is not None.

    ##    png (filepath, default=None) == [cwd]/cdgraf.png
    ##    res (int, default=300)  dpi
    """

    def plot_line(line, color='k', **kwargs):
        ax.plot([pos[0] / width for pos in line], [pos[1] / height for pos in line], color=color, **kwargs)

    def plot_text(x, y, s, *args, **kwargs):
        ax.text(x / width, y / height, s, *args, **kwargs)

####

#    result_copy = RankResult(**result._asdict())
#    result_copy = result_copy._replace(rankdf=result.rankdf.sort_values(by='meanrank'))
#    sorted_ranks, names, groups = get_sorted_rank_groups(result_copy, reverse)
#    cd = result.cd

    sorted_ranks, names, groups, sigd = get_sorted_rank_groups(cd, meanranks, reverse)

    if width is None:
        width = 6
####

    lowv = min(1, int(math.floor(min(sorted_ranks))))
    highv = max(len(sorted_ranks), int(math.ceil(max(sorted_ranks))))

    cline = 0.4
    textspace = 1
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            relative_rank = rank - lowv
        else:
            relative_rank = highv - rank
        return textspace + scalewidth / (highv - lowv) * relative_rank

    linesblank = 0.2 + 0.2 + (len(groups) - 1) * 0.1

    # add scale
    distanceh = 0.25
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((len(sorted_ranks) + 1) / 2) * 0.2 + minnotsignificant

    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    plot_line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        plot_line([(rankpos(a), cline - tick / 2),
                   (rankpos(a), cline)],
                  linewidth=0.7)

    for a in range(lowv, highv + 1):
        plot_text(rankpos(a), cline - tick / 2 - 0.05, str(a),
                  ha="center", va="bottom")

    for i in range(math.ceil(len(sorted_ranks) / 2)):
        chei = cline + minnotsignificant + i * 0.2
        plot_line([(rankpos(sorted_ranks[i]), cline),
                   (rankpos(sorted_ranks[i]), chei),
                   (textspace - 0.1, chei)],
                  linewidth=0.7)
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line([(rankpos(sorted_ranks[i]), cline),
                   (rankpos(sorted_ranks[i]), chei),
                   (textspace + scalewidth + 0.1, chei)],
                  linewidth=0.7)
        plot_text(textspace + scalewidth + 0.2, chei, names[i],
                  ha="left", va="center")

    # upper scale
    if not reverse:
        begin, end = rankpos(lowv), rankpos(lowv + cd)
    else:
        begin, end = rankpos(highv), rankpos(highv - cd)

    plot_line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
    plot_line([(begin, distanceh + bigtick / 2),
               (begin, distanceh - bigtick / 2)],
              linewidth=0.7)
    plot_line([(end, distanceh + bigtick / 2),
               (end, distanceh - bigtick / 2)],
              linewidth=0.7)
    plot_text((begin + end) / 2, distanceh - 0.05, "CD",
              ha="center", va="bottom")

    # no-significance lines
    side = 0.05
    no_sig_height = 0.1
    start = cline + 0.2
    for l, r in groups:
        plot_line([(rankpos(sorted_ranks[l]) - side, start),
                   (rankpos(sorted_ranks[r]) + side, start)],
                  linewidth=2.5)
        start += no_sig_height
####

#plt.savefig(imagePath)
    fp = 'cdgraf.png' if png is None else png
    fig.savefig(fp, dpi=res, bbox_inches="tight")
    plt.close(fig)
    return fp

####
#    return ax

