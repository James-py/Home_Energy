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

# %% Plot Original Data Line Graph

if False:
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    kW.drop(columns=[
        'DHWHP_Spy','Main_MTU']
        ).plot(ax=ax, colormap=cm.gist_rainbow)
    
    kW.Main_MTU.plot(color='k',linestyle='-', ax=ax)
    ax.legend()
    ax.set_ylabel('Power (kW)')
    ax.set_title('original data lines')

# %% Data Cleaning

for col in kW.columns:
    kW.loc[kW[col]<0, col] = np.nan
    kW.loc[kW[col]>1000, col] = np.nan

# %% Plot Original Data

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW.drop(columns=['K_Plg_2', 'K_Plg_4',
    'DHWHP_Spy','Main_MTU']
    ).plot.area(ax=ax, colormap=cm.gist_rainbow)

kW.Main_MTU.plot(color='k',linestyle='-', ax=ax)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('original data')

# %% Data Cleaning

kW.loc[:,'Main_MTU_Mod'] = kW.loc[:,'Main_MTU'] - 0.05
kW.loc[kW.Main_MTU_Mod<0, 'Main_MTU_Mod'] = np.nan
kW.loc[:,'Out_Plugs_Mod'] = kW.loc[:,'Out_Plugs'] #* 0.8
kW.loc[:,'Gar_Dryer_Mod'] = kW.loc[:,'Gar_Dryer'] #* 1.05
kW.loc[:,'K_Oven_Mod'] = kW.loc[:,'K_Oven'] #* 1.05
kW.loc[:,'Garage_Mod'] = kW.loc[:,'Garage'] * 1
kW.loc[:,'K_Fridge_Mod'] = kW.loc[:,'K_Fridge'] * 1
kW.loc[:,'Living_Rm_Mod'] = kW.loc[:,'Living_Rm'] #* 0.5  # was 0.75 on ECC
kW.loc[:,'Bed_G_Off_Mod'] = kW.loc[:,'Bed_G_Off'] #* 0.5  # was 0.75 on ECC
kW.loc[:,'K_DishW_Mod'] = kW.loc[:,'K_DishW'] #* 0.4
kW.loc[:,'Bed_Main_Mod'] = kW.loc[:,'Bed_Main'] #* 0.4
kW.loc[:,'DHWHP_Spy_Mod'] = kW.loc[:,'DHWHP_Spy'] * 0.71 # was 0.71 on ECC (both channels)
kW.sort_index(axis=1, inplace=True)

"""
NOTE: On Aug 25 at ~ 6PM I changed all the spyder multipliers on the ECC back to 1 or 2
I also added a second spyder legg for the double-pole of the Oven and for the Dishwasher
finished at ~6:26PM
"""


# %% 

"""
XKCD_COLORS
CSS4_COLORS
tab20
"""

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW.drop(columns=['K_Plg_2', 'K_Plg_4',
    'DHWHP_Spy','DHWHP_Spy_Mod', 'Main_MTU', 'Main_MTU_Mod', 'Gar_Dryer', 'K_Oven', 'Out_Plugs',
    'Garage', 'K_Fridge', 'Living_Rm', 'Bed_G_Off', 'K_DishW', 'Bed_Main']
    ).plot.area(ax=ax, colormap=cm.gist_rainbow)

kW.Main_MTU_Mod.plot(color='k',linestyle='-', ax=ax)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('cleaned data')

# %%

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW.loc[:,['DHWHP_Spy_Mod','DHW_MTU']].plot(ax=ax,alpha=0.35)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('DHW HP')

# %%

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW.loc[:,['K_Plg_1Is']].plot(ax=ax,alpha=0.35)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('Kitchen Plugs')

