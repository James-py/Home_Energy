#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 05:36:44 2023

@author: james
"""


# %% Import Libraries

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import cm

# %% update plotting data formats

plt.rcParams["date.autoformatter.hour"] = "%Y-%m-%d %H:%M"
plt.rcParams["date.autoformatter.minute"] = "%Y-%m-%d %H:%M"


# %% Turn off plotting of graphs that slow things down by setting to False
allplots = False
otherplots = False

# %% Plotting Functions


def simple_plot(df, caption, vertLine=1000, xlab='', ylab=''):
    fig, ax = plt.subplots(1, 1, tight_layout=True, num=caption)
    df.plot(ax=ax, marker='.', alpha=0.35, linestyle='None')
    if vertLine != 1000:
        plt.axvline(vertLine)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


def energy_plot(df, title, onOff=False):
    fig, ax = plt.subplots(1, 1, tight_layout=True, num=title)
    df.plot(ax=ax, marker='.', alpha=0.5, linestyle='None',
            legend=onOff)
    ax.set_ylabel("Daily Energy Consumed (kWh)")
    ax.set_xlabel("Date / Time")
    ax.set_title(title+'\n(n='+str(len(df.columns))+')')

# %% set up folder paths

# root folder for this project path object
CWD = Path(__file__).parent.resolve()
print('\nCWD Folder')
print(CWD)

# raw data folder path object
RAWDataPath = CWD.joinpath('Data')
print('\nSource Data (Raw) Folder')
print(RAWDataPath)

# %%


data_files = {}

for f in RAWDataPath.glob('*.csv'):
    data_files[f.stem] = f
    
    
# %%
col_names = [
    'Circuit',
    'Date_Time',
    'kW',
    'cost',
    'Voltage',
    'PF',
    ]
data_dict = {}

for n, f in data_files.items():
    data_dict[n] = pd.read_csv(f, names=col_names)
    
    
data_raw = pd.concat(data_dict)

# %%

# print(data.set_index('Date_Time', append=True))
data_WIP = data_raw.drop_duplicates(subset=['Circuit', 'Date_Time'])

# %%
# data_WIP.loc[:,'Date_Time'] = pd.to_datetime(data_WIP.Date_Time)


# %%
kW = data_WIP.pivot(index='Date_Time', columns='Circuit', values='kW')

kW.index = pd.to_datetime(kW.index)

kW.loc[kW.DHWHP_Spy<0, 'DHWHP_Spy'] = np.nan
# %% 

"""
XKCD_COLORS
CSS4_COLORS
tab20
"""

# cmap = mpl.color_sequences['tab10']

cmap = mcolors.CSS4_COLORS


# plot_df = kW.drop(columns=['DHWHP_Spy','Main_MTU'])
fig, ax = plt.subplots(1, 1, tight_layout=True)
kW.drop(columns=['DHWHP_Spy','Main_MTU']).plot.area(ax=ax, colormap=cm.gist_rainbow) # color=list(cmap))

kW.Main_MTU.plot(color='k',linestyle='-', ax=ax)
ax.legend()

