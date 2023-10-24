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
# import seaborn as sns
# import matplotlib as mpl
# import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import ListedColormap

# %% get today's date

today = datetime.datetime.now().date()
one_week_ago = today - datetime.timedelta(weeks=1)
start_date = datetime.date(2023,10,9)

# %% create my custom colourmap

"""
XKCD_COLORS
CSS4_COLORS
tab20
"""

cmap = ListedColormap(["darkorange", 
                       "forestgreen",
                       "slategrey",
                       "gold",
                       "lime",
                       "royalblue",
                       "lightcoral",
                       "lightgreen",
                       'blue',
                       'red',
                       'yellow',
                       "seagreen",
                       'fuchsia',
                       'cyan',
                       'indigo',
                       'olive',
                       'tan',
                       'skyblue',
                       'salmon',
                       'darkseagreen',
                       'limegreen',
                       'violet',
                       'darkblue',
                       'chocolate',
                       'silver',
                       'orange',
                       'deeppink'
                       ])


# %% close all figures

plt.close(fig='all')


# %% update plotting data formats

plt.rcParams["date.autoformatter.hour"] = "%Y-%m-%d %H:%M"
plt.rcParams["date.autoformatter.minute"] = "%Y-%m-%d %H:%M"
plt.rcParams["date.autoformatter.day"] = "%Y-%m-%d"

# %% Turn off plotting of graphs that slow things down by setting to False
allplots = False
otherplots = False

# %% Plotting Functions

def dots_plot(df, caption, vertLine=1000, xlab='Date / Time', ylab='',
                legend=False, start_date=one_week_ago):
    fig, ax = plt.subplots(1, 1)
    df.loc[start_date:,:].plot(ax=ax, colormap=cmap,
            marker='.', alpha=0.35, linestyle='None',legend=legend,
            x_compat=True)
    if vertLine != 1000:
        plt.axvline(vertLine)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(caption)

def area_plot(df, caption, drop_list=[], main_line=False, 
              legend=False, ylab='', start_date=one_week_ago):
    fig, ax = plt.subplots(1, 1)
    df.loc[start_date:,:].drop(columns=drop_list).plot.area(
        ax=ax, colormap=cmap, x_compat=True, legend=legend)
    if main_line:
        df.loc[start_date:,:].Main_MTU.plot(color='brown',linestyle='-', ax=ax)
        ax.legend()
    ax.set_ylabel(ylab)
    ax.set_xlabel("Date / Time")
    ax.set_title(caption)

def lines_plot(df, caption, drop_list=[], legend=True, ylab='', start_date=one_week_ago):
    fig, ax = plt.subplots(1, 1)
    df.loc[start_date:,:].drop(columns=drop_list).plot(alpha=0.75,
        ax=ax, colormap=cmap, legend=legend, x_compat=True)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Date / Time")
    ax.set_title(caption)
    ax.axhline(color='k')

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

# %% Pivot and set datetime index
kW = data_WIP.pivot(index='Date_Time', columns='Circuit', values='kW')
kW.index = pd.to_datetime(kW.index)


# %% Data Cleaning - filter noise

for col in kW.columns:
    kW.loc[kW[col]<0, col] = np.nan
    kW.loc[kW[col]>12, col] = np.nan

# %% Make a copy of the kW data for data cleaning
# Drop duplicate Oven data

kW_mod = kW.copy()
kW_mod.drop(columns=['K_Oven1','K_Oven2'], inplace=True)
kW_mod.drop(columns=['DHWHP_Sp1','DHWHP_Sp2'], inplace=True)
kW_mod.drop(columns=['K_DishW_1','K_DishW_2'], inplace=True)
kW_mod.drop(columns=['Gar_Dry1','Gar_Dry2'], inplace=True)

# %% Data Cleaning - Spyder Leg Calibration

calibration_dict = {
    'K_Plg_4Ts':1.04,
    'K_DishW':0.55,
    'Gar_Dryer':1.04,
    'K_Oven':1.04,
    'Out_Plugs':0.75,
    'Garage':0.85,
    'Freezer':0.85,
    'K_Fridge':0.85,
    'Living_Rm':0.65, 
    'Bed_G_Off':0.75, 
    'Bed_Main':0.60,
    'DHWHP_Spy':0.71, 
    'K_Plg_2MW':0.75,
    'Heat_Beds':1.0,
    'Heat_LvRm':1.0
    }

for circuit, multiplier in calibration_dict.items():
    kW_mod.loc[:,circuit] = kW.loc[:,circuit] * multiplier
    
kW_mod.sort_index(axis=1, inplace=True)


"""
NOTE: On Aug 25 at ~ 6PM I changed all the spyder multipliers on the ECC back to 1 or 2
I also added a second spyder legg for the double-pole of the Oven and for the Dishwasher
finished at ~6:26PM
"""

# %% Fix Living Room Heating


def fix_living_room_heating(df_mod, df, circuit, cutoff, offset, 
                            multiplier, reduction_multiplier):
    df_mod.loc[(df[circuit]<=cutoff) & (df[circuit]>=offset),circuit] = df.loc[
        (df[circuit]<=cutoff) & (df[circuit]>=offset),circuit] - offset
    df_mod.loc[(df_mod[circuit]<offset),circuit] = df.loc[
        (df[circuit]<offset),circuit] * reduction_multiplier
    df_mod.loc[df[circuit]>cutoff,circuit] = df.loc[
        df[circuit]>cutoff,circuit] * multiplier
        
    
fix_living_room_heating(kW_mod, kW, 'Heat_LvRm', 0.6, 0.1, 1.12, 0.5)

# kW_mod.loc[(kW.Heat_LvRm<0.5),'Heat_LvRm'] = kW.loc[
#     (kW.Heat_LvRm<0.5),'Heat_LvRm'] * 0.5

# kW_mod.loc[kW.Heat_LvRm>0.5,'Heat_LvRm'] = kW.loc[kW.Heat_LvRm>0.5,'Heat_LvRm'] * 1

# area_plot(kW_mod, 'Power Data Mod - Area', main_line=True,
#           drop_list=['DHW_MTU','Main_MTU','Test_MTU'], 
#           legend=True, ylab='Power (kW)')


# %% Fix Heating


def fix_bed_heating(df_mod, df, circuit, cutoff, offset, multiplier):
    df_mod.loc[(df[circuit]<cutoff) & (df[circuit]>offset),circuit] = df.loc[
        (df[circuit]<cutoff) & (df[circuit]>offset),circuit] - offset
    df_mod.loc[df[circuit]>cutoff,circuit] = df.loc[
        df[circuit]>cutoff,circuit] * multiplier
        
    
fix_bed_heating(kW_mod, kW, 'Heat_Beds', 0.5, 0.1, 1.12)


# %% Data Cleaning - double pole circuits
'''
some double pole circuits are symmetrical, so I don't need to use two spyder CTs
I started with two, and then changed to using just one CT
That meant changing the multiplier on the MTU and creating extra channels of 
data while I was checking things
created some irregular data
This section fixes that
- combines the extra data channels back into one and drops the extra channels

'''

def fix_double_pole(df_mod, df, circuit, c1, c2, multiplier):
    df_mod.loc[df[c1]>0,circuit] = df.loc[
        df[c1]>0,[c1,c2]].sum(axis=1) * multiplier
    df_mod[circuit].fillna(0, inplace=True)

# K_Oven
fix_double_pole(kW_mod, kW, 'K_Oven', 'K_Oven1','K_Oven2', 1.04)

# DHW Heat Pump
fix_double_pole(kW_mod, kW, 'DHWHP_Spy', 'DHWHP_Sp1','DHWHP_Sp2', 0.71)

# Dishwasher
fix_double_pole(kW_mod, kW, 'K_DishW', 'K_DishW_2','K_DishW_1', 0.55)

# Laundry Dryer
fix_double_pole(kW_mod, kW, 'Gar_Dryer', 'Gar_Dry1','Gar_Dry2', 1.04)


# %% Fill Missing Power Load
# spyder CTs are less accurate and don't reliably detect loads <100 W
# this fills in a minimum load when ct dectects less than the minimum


def fill_one_circuit(df_mod, df, circuit, load):
    df_mod.loc[df[circuit]<load,circuit] = load
    
# Living Room: 30 W appears to be the router that is always on    
fill_one_circuit(kW_mod, kW, 'Living_Rm', 0.035)
# various standby and chargers in bedrooms are still a small load
fill_one_circuit(kW_mod, kW, 'Bed_Main', 0.01)
# raspberry pi and doc
fill_one_circuit(kW_mod, kW, 'Bed_G_Off', 0.02)
# When freezer is on, the spyder seems to underestimate the load sometimes
kW_mod.loc[kW.Freezer.between(0.01,0.03), 'Freezer'] = 0.05


# %% Use Test MTU for Freezer for the week it was on that circuit

kW_mod.loc['2023-09-30 12:00':'2023-10-10 09:50','Freezer'] = kW_mod.loc[
    '2023-09-30 12:00':'2023-10-10 09:50','Test_MTU']

# %% Total Power Comparison Calcs
# - uses spyder data for DHW
# fill Main MTU data using spyder sum

kW_tot_compare = pd.DataFrame(kW_mod['Main_MTU'])
kW_tot_compare['Spy_Sum'] = kW_mod.drop(columns=['DHW_MTU','Main_MTU','Test_MTU']
                                        ).sum(axis=1)
kW_tot_compare['Main_MTU'] = kW_tot_compare['Main_MTU'].fillna(kW_tot_compare.Spy_Sum)

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
BCH_data_WIP = BCH_data_raw.reset_index().drop_duplicates(subset=['Account Number','Interval Start Date/Time'])
BCH_data_WIP['Date_Time'] = pd.to_datetime(BCH_data_WIP['Interval Start Date/Time']).dt.floor('Min')

# %% create df with only BCH energy and set datetime index
BCH_kWh = BCH_data_WIP.pivot(index='Date_Time', 
                             columns='Account Number', 
                             values='Net Consumption (kWh)')

#%% calculate hourly energy 

kWh = kW_mod.resample('1H').mean()

# %% Total Energy Compare and combine TED and BCH data

kWh_tot_compare = pd.DataFrame(kWh[['Main_MTU', 'DHWHP_Spy']])
kWh_tot_compare['Spy_Sum'] = kWh.drop(columns=['DHW_MTU','Main_MTU','Test_MTU']).sum(axis=1)
kWh_tot_compare['Main_MTU'] = kWh['Main_MTU'].fillna(kWh_tot_compare.Spy_Sum)
kWh_tot_compare['MTU_Spy_Diff'] = kWh_tot_compare.Main_MTU - kWh_tot_compare.Spy_Sum

# kWh_tot_compare.loc[kWh_tot_compare['MTU_Spy_Diff']<0, 'MTU_Spy_Diff'] = np.nan
kWh_tot_compare['BCH'] = BCH_kWh[12014857]
kWh_tot_compare['MTU_BCH_Diff'] = kWh_tot_compare.Main_MTU - kWh_tot_compare.BCH

# %% Daily Energy Totals

kWh_daily = kWh_tot_compare.resample('1d').sum()

kWh_daily['MTU_Spy_Diff_pct'] = kWh_daily.MTU_Spy_Diff / kWh_daily.Spy_Sum *100
kWh_daily['MTU_BCH_Diff_pct'] = kWh_daily.MTU_BCH_Diff / kWh_daily.BCH *100


print('\n', "Average daily DHW HP Energy Consumption:", 
      round(kWh_daily.DHWHP_Spy['2023-09-01':].mean(),2), "kWh",'\n')

# print table
print(kWh_daily)

# %% bar plot of daily total energy

fig, ax = plt.subplots(1, 1)
kWh_daily.loc[start_date:,
              ['Main_MTU', 'Spy_Sum', 'BCH']
              ].plot.bar(legend=True, ax=ax)
ax.set_xticklabels(kWh_daily[start_date:].index.strftime('%Y-%m-%d'))
ax.axhline(color='k')
ax.set_ylabel('Daily Energy Consumption (kWh)')
ax.set_xlabel('Date')
fig.tight_layout()
    

# %% Plot Hourly Energy Total Comparison

lines_plot(kWh_tot_compare, 'MTU and Spyder Total Energy Comparison', 
           drop_list=['DHWHP_Spy'], legend=True, ylab='Hourly Energy (kWh)')

# %% Plot Cleaned Power Data Area All

area_plot(kW_mod, 'Power Data Mod - Area', main_line=True,
          drop_list=['DHW_MTU','Main_MTU','Test_MTU'], 
          legend=True, ylab='Power (kW)') #, start_date=start_date)

# %% Plot Power Total Comparison

lines_plot(kW_tot_compare, 'MTU and Spyder Total Power Comparison', 
           drop_list=[], legend=True, ylab='Power (kW)')


# %% Plot Original Data Area Graph

area_plot(kW, 'Original Power Data - Area', main_line=True,
          drop_list=['DHW_MTU','Main_MTU','Test_MTU'], 
          legend=True, ylab='Power (kW)')


# %% Plot Hourly Energy Area all channels

area_plot(kWh, 'Hourly Energy - Area', main_line=True,
          drop_list=['DHW_MTU','Main_MTU','Test_MTU'], 
          legend=True, ylab='Hourly Energy (kWh)')


# %% 

# lines_plot(kW_mod[['Gar_Dryer','Test_MTU']].resample('1H').mean(), 'Dryer', ylab='hourly energy kWh',
#           legend=True)


# lines_plot(kW.filter(like='DishW').resample('1H').mean(), 'Dishwasher', ylab='hourly energy kWh',
#           legend=True)

# lines_plot(kW[['Freezer','Test_MTU']], 'Freezer original power', ylab='kW',
#           legend=True)

lines_plot(kW_mod.loc['2023-10-10 10:00':,['Heat_Beds','Test_MTU']], 'Heat_Beds kW 3', ylab='kW',
          legend=True)

# # %% One-off area Plot

# area_plot(kW_mod.filter(like='Oven'), 'Oven', ylab='power kW',
#           legend=True)

# # %% One-off dots Plot

# dots_plot(kW_mod[['Living_Rm','Test_MTU']], 'Living Room Mod', ylab='power kW',
#           legend=True)

# %% One-off dots Plot

# dots_plot(kW_mod[['Bed_Main','Test_MTU']], 'Bed Room Mod', ylab='power kW',
#           legend=True)

# # %% One-off dots Plot

# dots_plot(kW_mod[['K_DishW','Test_MTU']], 'Dishwasher', ylab='power kW',
#           legend=True)


# # %% Plot Original Data Line Graph Spyders only

# lines_plot(kW, 'original spyder data - lines', 
#            drop_list=['Main_MTU', 'DHW_MTU','Test_MTU'], 
#            legend=True, ylab='Power (kW)')

# dots_plot(kW_mod[['Freezer','Test_MTU']], 'Freezer Mod', ylab='power kW',
#           legend=True)