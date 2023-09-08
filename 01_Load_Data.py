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
# import datetime
# import seaborn as sns
# import matplotlib as mpl
# import matplotlib.colors as mcolors
from matplotlib import cm

plt.close(fig='all')


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

# %% get file paths to all the csvs

data_files = {}

for f in RAWDataPath.glob('*.csv'):
    data_files[f.stem] = f
    
# %% Read csvs and store as DataFrames in a dictionary

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


# %% Combine data into one dataframe and drop duplicates
    
data_raw = pd.concat(data_dict)
data_WIP = data_raw.drop_duplicates(subset=['Circuit', 'Date_Time'])
# not efficient to parse datetime here
# data_WIP.loc[:,'Date_Time'] = pd.to_datetime(data_WIP.Date_Time)


# %% Pivot and set datetime index
kW = data_WIP.pivot(index='Date_Time', columns='Circuit', values='kW')
kW.index = pd.to_datetime(kW.index)

# %% Plot Original Data Line Graph Spyders only

if False:
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    kW.drop(columns=['Main_MTU', 'DHW_MTU']
        ).plot(ax=ax, colormap=cm.gist_rainbow, x_compat=True)
    ax.legend()
    ax.set_ylabel('Power (kW)')
    ax.set_title('original spyder data - lines')

# %% Data Cleaning - filter noise

for col in kW.columns:
    kW.loc[kW[col]<0, col] = np.nan
    kW.loc[kW[col]>1000, col] = np.nan

# %% Plot Original Data Area Graph

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW.drop(columns=['DHW_MTU','Main_MTU']
    ).plot.area(ax=ax, colormap=cm.gist_rainbow, x_compat=True)

kW.Main_MTU.plot(color='k',linestyle='-', ax=ax)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('Original Power Data - Area')

# %% Data Cleaning - Spyder Leg Calibration

kW_mod = kW.copy()

kW_mod.loc[:,'K_Plg_4Ts'] = kW.loc[:,'K_Plg_4Ts'] * 1.08
kW_mod.loc[:,'Gar_Dryer'] = kW.loc[:,'Gar_Dryer'] * 1.08
kW_mod.loc[:,'K_Oven'] = kW.loc[:,'K_Oven'] * 1.05
kW_mod.loc[:,'Out_Plugs'] = kW.loc[:,'Out_Plugs'] * 0.55
# kW_mod.loc[:,'K_Fridge'] = kW.loc[:,'K_Fridge'] * 1
kW_mod.loc[:,'Living_Rm'] = kW.loc[:,'Living_Rm'] * 0.75  # was 0.75 on ECC
kW_mod.loc[:,'Bed_G_Off'] = kW.loc[:,'Bed_G_Off'] * 0.85  # was 0.75 on ECC
kW_mod.loc[:,'K_DishW'] = kW.loc[:,'K_DishW'] * 0.75
kW_mod.loc[:,'Bed_Main'] = kW.loc[:,'Bed_Main'] * 0.85
kW_mod.loc[:,'DHWHP_Spy'] = kW.loc[:,'DHWHP_Spy'] * 0.71 # was 0.71 on ECC (both channels)
kW_mod.sort_index(axis=1, inplace=True)

"""
NOTE: On Aug 25 at ~ 6PM I changed all the spyder multipliers on the ECC back to 1 or 2
I also added a second spyder legg for the double-pole of the Oven and for the Dishwasher
finished at ~6:26PM
"""


# %% Cleaned Power Data Area Plot

"""
XKCD_COLORS
CSS4_COLORS
tab20
"""

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW_mod.drop(columns=['DHW_MTU','Main_MTU']
    ).plot.area(ax=ax, colormap=cm.gist_rainbow, x_compat=True)

kW_mod.Main_MTU.plot(color='k',linestyle='-', ax=ax)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('Cleaned Power Data - Area')


# %% DHW Spyder Vs MTU Comparison

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW_mod.loc[:,['DHWHP_Spy','DHW_MTU']].plot(ax=ax,alpha=0.35, x_compat=True)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('DHW HP Spyder vs MTU')


# %% Total Power Comparison 
# - uses spyder data for DHW
# fill Main MTU data using spyder sum

kW_tot_compare = pd.DataFrame(kW_mod['Main_MTU'])
kW_tot_compare['Spy_Sum'] = kW_mod.drop(columns=['DHW_MTU','Main_MTU']).sum(axis=1)
kW_tot_compare['Main_MTU'] = kW_tot_compare['Main_MTU'].fillna(kW_tot_compare.Spy_Sum)

# %% Plot Power Total Comparison

fig, ax = plt.subplots(1, 1, tight_layout=True)
kW_tot_compare.plot(ax=ax,alpha=0.75, x_compat=True)
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_title('Compare totals')


# %% import BC Hydro data

BCH_data_files = {}

for f in RAWDataPath.joinpath('BC_Hydro').glob('*.csv'):
    BCH_data_files[f.stem] = f


# BCH_col_names = [
#     'Circuit',
#     'Date_Time',
#     'kW',
#     'cost',
#     'Voltage',
#     'PF',
#     ]
BCH_data_dict = {}

for n, f in BCH_data_files.items():
    BCH_data_dict[n] = pd.read_csv(f)
    
# %% Combine BCH data into one dataframe and drop duplicates

'''
['Account Number', 'Interval Start Date/Time', 'Net Consumption (kWh)',
       'Demand (kW)', 'Power Factor (%)']
'''

BCH_data_raw = pd.concat(BCH_data_dict)
BCH_data_WIP = BCH_data_raw.drop_duplicates(subset=['Interval Start Date/Time'])
BCH_data_WIP['Date_Time'] = pd.to_datetime(BCH_data_WIP['Interval Start Date/Time'])
BCH_data_WIP['Date_Time'] = BCH_data_WIP.Date_Time.dt.floor('Min')

# %% create df with only energy and set datetime index
BCH_kWh = BCH_data_WIP.pivot(index='Date_Time', 
                             columns='Account Number', 
                             values='Net Consumption (kWh)')
BCH_kWh.index = pd.to_datetime(BCH_kWh.index)


#%% calculate hourly energy and combine TED and BCH data

kWh = kW_tot_compare.resample('1H').mean()
kWh['MTU_Spy_Diff'] = kWh.Main_MTU - kWh.Spy_Sum
kWh.loc[kWh['MTU_Spy_Diff']<0, 'MTU_Spy_Diff'] = np.nan
kWh['BCH'] = BCH_kWh[12014857]
kWh['MTU_BCH_Diff'] = kWh.Main_MTU - kWh.BCH

# %% Plot hourly Energy data

fig, ax = plt.subplots(1, 1, tight_layout=True)
kWh.plot(ax=ax,alpha=0.75, x_compat=True)
ax.legend()
ax.set_ylabel('Energy (kWh)')
ax.set_title('Compare totals - Energy')
ax.axhline(color='k')

# %% print table
print(kWh.resample('1d').sum())

