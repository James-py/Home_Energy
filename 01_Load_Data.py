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
plt.rcParams["date.autoformatter.day"] = "%Y-%m-%d"

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
RAWDataPath = CWD.joinpath('Data_Raw')
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


# %% Pivot and set datetime index
kW = data_WIP.pivot(index='Date_Time', columns='Circuit', values='kW')

kW.index = pd.to_datetime(kW.index)

# %%


fig, ax = plt.subplots(1, 1, tight_layout=True)
kW.drop(columns=[
    'DHWHP_Spy','Main_MTU']
    ).plot.area(ax=ax, colormap=cm.gist_rainbow)

kW.Main_MTU.plot(color='k',linestyle='-', ax=ax)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('original data')

# %% Data Cleaning
kW.loc[kW.DHWHP_Spy<0, 'DHWHP_Spy'] = np.nan
kW.loc[:,'Gar_Dryer_Mod'] = kW.loc[:,'Gar_Dryer'] * 1.05
kW.loc[:,'K_Oven_Mod'] = kW.loc[:,'K_Oven'] * 1.05
kW.loc[:,'Garage_Mod'] = kW.loc[:,'Garage'] * 0.6
kW.loc[:,'K_Fridge_Mod'] = kW.loc[:,'K_Fridge'] * 0.9
kW.loc[:,'Living_Rm_Mod'] = kW.loc[:,'Living_Rm'] * 0.8
kW.loc[:,'Bed_G_Off_Mod'] = kW.loc[:,'Bed_G_Off'] * 0.5
kW.loc[:,'K_DishW_Mod'] = kW.loc[:,'K_DishW'] * 0.4
kW.loc[:,'DHW_MTU_Mod'] = kW.loc[:,'DHW_MTU'] * 1.0
kW.sort_index(axis=1, inplace=True)
# %% 

"""
XKCD_COLORS
CSS4_COLORS
tab20
"""

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW.drop(columns=[
    'DHWHP_Spy','Main_MTU', 'Gar_Dryer', 'K_Oven', 'DHW_MTU',
    'Garage', 'K_Fridge', 'Living_Rm', 'Bed_G_Off', 'K_DishW']
    ).plot.area(ax=ax, colormap=cm.gist_rainbow)

kW.Main_MTU.plot(color='k',linestyle='-', ax=ax)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('cleaned data')