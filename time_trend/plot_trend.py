#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:56:36 2021
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
# colors = plt.cm.viridis(np.linspace(0,1,7))

hfont = {'fontname':'times'}
plt.rcParams["mathtext.fontset"] = "cm"
from mpl_axes_aligner import align
from matplotlib.ticker import MultipleLocator
import pandas as pd


ls_book = {'densely dotted': (0, (1, 1)),
            'densely dashed': (0, (5, 1)),
             "densely dashdotted": (0, (3, 1, 1, 1)),
             'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}


def plot_linestyles(ax, linestyles, title):
    X, Y = np.linspace(0, 100, 10), np.zeros(10)
    yticklabels = []

    for i, (name, linestyle) in enumerate(linestyles):
        ax.plot(X, Y+i, linestyle=linestyle, linewidth=1.5, color='black')
        yticklabels.append(name)

    ax.set_title(title)
    ax.set(ylim=(-0.5, len(linestyles)-0.5),
           yticks=np.arange(len(linestyles)),
           yticklabels=yticklabels)
    ax.tick_params(left=False, bottom=False, labelbottom=False)
    ax.spines[:].set_visible(False)

    # For each line style, add a text annotation with a small offset from
    # the reference point (0 in Axes coords, y tick value in Data coords).
    for i, (name, linestyle) in enumerate(linestyles):
        ax.annotate(repr(linestyle),
                    xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
                    xytext=(-6, -12), textcoords='offset points',
                    color="blue", fontsize=8, ha="right", family="monospace")


# df = np.genfromtxt('final_stats(3 countries).csv', delimiter=',', skip_header=True)
# df = np.transpose(df)
df = pd.read_csv("final_stats.csv")
week_strings = list(df.start_date)
# # read week strings separately to avoid type error
# week_strings = np.genfromtxt('final_stats(3 countries).csv', 
#                               delimiter=',', skip_header=True, dtype='str')
# week_strings = np.transpose(week_strings)
# week_strings = week_strings[-2]

# begin plotting
plt.set_cmap("Set1");
fig, ax1 = plt.subplots(1, 1, figsize=(25, 10))

lw = 4
ax1.plot(df.week,  df["molnupiravir"], linewidth=lw, linestyle = ls_book["densely dashdotdotted"], label='Molnupiravir')
ax1.plot(df.week, df["remdesivir"], linewidth=lw, linestyle = ls_book["densely dashed"],label='Remdesivir')
ax1.plot(df.week, df["hcq"], linewidth=lw, linestyle = ls_book["densely dashdotted"], label='Hydroxychloroquine')
ax1.plot(df.week, df["ivermectin"],linewidth=lw, linestyle = ls_book["densely dotted"], label='Ivermectin')


# plot the ratio of tweets mentioned a certain drug on the left y axis

# ax1.fill_between(df.week,  df["molnupiravir"], 0, alpha=0.5, color=colors[4], label='Molnupiravir')
# ax1.fill_between(df.week, df["molnupiravir"] + df["remdesivir"], df["molnupiravir"], alpha=0.5, color=colors[0], label='Remdesivir')
# ax1.fill_between(df.week, df["hcq"] + df["molnupiravir"] + df["remdesivir"], df["molnupiravir"] + df["remdesivir"], alpha=0.5, color=colors[6], label='Hydroxychloroquine')

# ax1.fill_between(df.week, df["hcq"] + df["molnupiravir"] + df["remdesivir"] + df["ivermectin"], df["hcq"] + df["molnupiravir"] + df["remdesivir"], alpha=0.5, color=colors[2],label='Ivermectin')

weeks_shown = [int(min(df.week))] + list(np.arange(9, 98, 8)) + [int(max(df.week))]
weeks_shown = list(map(lambda x: int(x), weeks_shown))
ax1.set_xticks(weeks_shown)
ax1.set_xticklabels([week_strings[index-2] for index in weeks_shown],
                    rotation=15)

# plot the new cases of each countray on the right y axis
ax2 = ax1.twinx()
ax2.plot(df.week, df["new"]/1.0e6, alpha=0.65,label='US weekly', linewidth=lw)

# ax2.step(df.week, df["new"]/1.0e6, where='mid', label='US weekly', linewidth=2)
# ax2.step(df.week, df["UK.new"]/1.0e6, where='mid', label='UK', linewidth=2, alpha=0.7, color=colors[3])
# ax2.step(df.week, df["Canada.new"]/1.0e6, where='mid', label='Canada', linewidth=2, alpha=0.7, color=colors[5])
# ax2.step(df.week, df["India.new"]/1.0e6, where='mid', label='India', color='#de1259')
# ax2.step(df.week, df["Philippines.new"]/1.0e6, where='mid', label='Philippines', color='#FFC300')

# ax2.plot(df.week, (df["US.new"]+df["UK.new"]+df["Canada.new"]+df["India.new"]+df["Philippines.new"])/1.0e6, where='mid', 
#          label='Total', color='#9e9e9e')
# ax2.plot(df.week, df["US.new"]/1.0e6, label='US', color='#f56a07')
# ax2.plot(df.week, df["UK.new"]/1.0e6, label='UK', color='#1010e6')
# ax2.plot(df.week, df["Canada.new"]/1.0e6, label='Canada', color='#B125D5')
# ax2.plot(df.week, df["India.new"]/1.0e6, label='India', color='#94C510')
# ax2.plot(df.week, df["Philippines.new"]/1.0e6, label='Philippines', color='#de1259')

# ax2.plot(df.week, (df["US.new"]+df["UK.new"]+df["Canada.new"]+df["India.new"]+df["Philippines.new"])/1.0e6, label='Total', color='#9e9e9e')
# ax2.step(df.week, (df["US.new"]+df["UK.new"]+df["Canada.new"])/1.0e6, where='mid', label='Total', color='#9e9e9e')

ax2.tick_params(axis='both', which='both', labelsize=18)

# set x and y axis limits and labels
ax1.set_xlim([min(df.week), max(df.week)])
ax1.tick_params(axis='both', which='both', labelsize=18, length=4)
ax1.tick_params(axis='x', which='major', labelsize=15, width=3, length=6)
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax2.set_ylim([0, 18])
ax1.set_ylim([0, 52])

# Adjust the plotting range of two y axes
org1 = 0.0  # Origin of first axis
org2 = 0.0  # Origin of second axis
pos = 1.0e-5  # Position the two origins are aligned
align.yaxes(ax1, org1, ax2, org2, pos)

ax1.set_ylabel('Number of Drug-related Tweets\n(Thousand)', fontsize=20)
ax2.set_ylabel('New Cases (Million)', fontsize=20)

# show legends
ax1.legend(fontsize=15, loc='upper left')
ax2.legend(fontsize=15, loc='upper right')

# save figures
# plt.savefig('nene_covid_plot.svg', bbox_inches='tight')
plt.savefig('trend_analysis.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()
