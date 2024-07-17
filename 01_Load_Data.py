"""
Created on Thu Aug 24 05:36:44 2023
@author: james

magic command for plots:
%matplotlib

terminal command to get data from Raspberry Pi
rsync -a james@rpi.local:/home/james/Data/ ~/repo/Home_Energy/Data_Raw

Use wget in Terminal to get Environment Canada Weather Data
for year in `seq 2023 2024`;do for month in `seq 1 12`;do wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=52941&Year=${year}&Month=${month}&Day=14&timeframe=1&submit= Download+Data" ;done;done

"""

# %% Import Libraries

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_axes_aligner import align

# %% get today's date

today = datetime.datetime.now().date()
one_week_ago = today - datetime.timedelta(weeks=1)
four_weeks_ago = today - datetime.timedelta(weeks=4)
start_date = four_weeks_ago  # datetime.date(2023,10,9)

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

cmap2 = ListedColormap([
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
        df.loc[start_date:,:].Main_MTU.plot(color='k',linestyle='-', ax=ax)
        ax.legend()
    ax.set_ylabel(ylab)
    ax.set_xlabel("Date / Time")
    ax.set_title(caption)

def lines_plot(df, df2, caption, drop_list=[], legend=True, ylab='', 
               start_date=one_week_ago, 
               vs_temp=False, col2 ='Temp (°C)'):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    df.loc[start_date:,:].drop(columns=drop_list).plot(alpha=0.75,
        ax=ax, colormap=cmap2, legend=legend, x_compat=True)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Date / Time")
    ax.set_title(caption)
    ax.axhline(color='k')
    # y1_max = ax.get_ybound()[1]
    # nticks = 11
    if vs_temp:
        ax2 = ax.twinx()
        colour2='tab:blue'
        ax2.plot(df2.loc[start_date:,col2], color=colour2, linestyle='--')
        ax2.set_ylabel('Temperature (°C)', color=colour2)
        ax2.tick_params(axis='y', labelcolor=colour2)
        align.yaxes(ax, 0, ax2, 0, 0.2)        
        
# plt.close('all')
# lines_plot(kWh_tot_compare,weather_df, 'MTU and BCH Total Energy Comparison',
#            drop_list=['DHWHP_Spy', 'Spy_Sum','MTU_Spy_Diff'], legend=True, ylab='Hourly Energy (kWh)', 
#            start_date=start_date, vs_temp=True)


# %% set up folder paths

# root folder for this project path object
CWD = Path(__file__).parent.resolve()
print('\nCWD Folder')
print(CWD)

# raw data folder path object
RAWDataPath = CWD.joinpath('Data_Raw')
print('\nSource Data (Raw) Folder')
print(RAWDataPath)

# BC Hydro data folder path object
BCH_Path = CWD.joinpath('Data_BC_Hydro')
print('\nBC Hydro Data Folder')
print(BCH_Path)

# Weather data folder path object
Weather_Path = CWD.joinpath('Data_Weather')
print('\nWeather data Folder')
print(Weather_Path)


# %% get file paths to all the csvs
def get_filepaths(f_path, f_pattern):
    file_dict = {}
    for f in f_path.glob(f_pattern):
        file_dict[f.stem] = f
    return file_dict

TED_files = get_filepaths(RAWDataPath, '*.csv')
Weather_files = get_filepaths(Weather_Path, '*.csv')

# %%
# read Data

def read_data(fp_dict, use_TED_cols=False):
    col_names = [
        'Circuit',
        'Date_Time',
        'kW',
        'cost',
        'Voltage',
        'PF',
        ]
    data_dict = {}
    for n, f in fp_dict.items():
        if use_TED_cols:
            data_dict[n] = pd.read_csv(f, names=col_names)
        else:
            data_dict[n] = pd.read_csv(f)
    return pd.concat(data_dict)


weather_data_df = read_data(Weather_files)
TED_data_df = read_data(TED_files, use_TED_cols=True)  

# %%

weather_df = weather_data_df.set_index('Date/Time (LST)')
weather_df.index = pd.to_datetime(weather_df.index)
weather_df.sort_index(inplace=True)
# %%  drop duplicates
    
data_WIP = TED_data_df.drop_duplicates(subset=['Circuit', 'Date_Time'], keep='last')

# %% Pivot and set datetime index
kW = data_WIP.pivot(index='Date_Time', columns='Circuit', values='kW')
kW.index = pd.to_datetime(kW.index)
kW.sort_index(inplace=True)

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
    'Gar_Dryer':1,
    'K_Oven':1.1,
    'Out_Plugs':0.7,
    'Garage':0.8,
    'Freezer':0.85,
    'K_Fridge':0.85,
    'Living_Rm':0.6, 
    'Bed_G_Off':0.6, 
    'Bed_Main':0.60,
    'DHWHP_Spy':0.7, 
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

# %% Fix  Heating


def fix_heating(df_mod, df, circuit, cutoff, offset, 
                            multiplier1, spike):
    # when signal is > cutoff
    rows = df[circuit]>cutoff
    df_mod.loc[rows,circuit] = (df.loc[rows,circuit]
        + (df.loc[rows,circuit] - cutoff) * multiplier1)
    # when signal is <= cutoff and >= offset
    # with a 0.05 W deadband
    rows = (df[circuit]<=(cutoff-0.05)) & (df[circuit]>=offset)
    df_mod.loc[rows,circuit] = (df.loc[rows,circuit]
        - offset)
    # when signal is < offset
    df_mod.loc[(df[circuit]<offset),circuit] = 0
    # when signal is > spike
    rows = df[circuit]>spike
    df_mod.loc[rows,circuit] = spike-0.5
        
    
fix_heating(kW_mod, kW, 'Heat_LvRm', 0.75, 0.1, 0.25, 3.0)

fix_heating(kW_mod, kW, 'Heat_Beds', 0.45, 0.1, 0.25, 3.0)

# %% Heat Fix testing

# Plots for testing
# Bed Rooms
# lines_plot(kW.loc['2023-10-10 10:00':'2023-10-28 13:20',['Heat_Beds','Test_MTU']], 
#            'Heat_Beds kW Oct 10 to 28', ylab='kW',
#           legend=True)
# lines_plot(kW_mod.loc[:'2023-10-28 13:20',['Heat_Beds','Test_MTU']], 
#             'Heat_Beds kW_mod Oct 10 to 28', ylab='kW',
#             start_date='2023-10-10', legend=True)
# # Living Room
# lines_plot(kW.loc['2023-10-28 13:30':,['Heat_LvRm','Test_MTU']], 
#             'Heat Living Room kW Oct 28', ylab='kW',
#           legend=True)
# lines_plot(kW_mod.loc[:,['Heat_LvRm','Test_MTU']], 
#             'Heat Living Room kW_mod Oct 28', ylab='kW',
#             start_date='2023-10-28', legend=True)

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
    df_mod[circuit] = df_mod[circuit].fillna(0)

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


# DHW_HP: 10 W appears to be a standby load   
fill_one_circuit(kW_mod, kW, 'DHWHP_Spy', 0.01)    
# Living Room: 30 W appears to be the router that is always on    
fill_one_circuit(kW_mod, kW, 'Living_Rm', 0.01)
# various standby and chargers in bedrooms are still a small load
fill_one_circuit(kW_mod, kW, 'Bed_Main', 0.01)
# raspberry pi and doc
fill_one_circuit(kW_mod, kW, 'Bed_G_Off', 0.01)
# When freezer is on, the spyder seems to underestimate the load sometimes
kW_mod.loc[kW.Freezer.between(0.01,0.03), 'Freezer'] = 0.05

# %% Total Power Comparison Calcs
# - uses spyder data for DHW
# fill Main MTU data using spyder sum

kW_tot_compare = pd.DataFrame(kW_mod['Main_MTU'])
kW_tot_compare['Spy_Sum'] = kW_mod.drop(columns=['DHW_MTU','Main_MTU','Test_MTU']
                                        ).sum(axis=1)
kW_tot_compare['Main_MTU'] = kW_tot_compare['Main_MTU'].fillna(kW_tot_compare.Spy_Sum)

# %% import BC Hydro data

BCH_data_files = {}

for f in BCH_Path.glob('*.csv'):
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
    

# %%

for n, df in BCH_data_dict.items():
    print(n)
    df['Date_Time'] = pd.to_datetime(df['Interval Start Date/Time'],yearfirst=True).dt.floor('Min')
    if df['Account Number'].dtype == 'O':
        df['Account Number'] = pd.to_numeric(df['Account Number'].str.removeprefix("'"))
    display(df.head(3))
    print('\n')
# %% 
# Combine BCH data into one dataframe and drop duplicates


# df[['Account Number', 'Interval Start Date/Time', 'Net Consumption (kWh)',
#        'Demand (kW)', 'Power Factor (%)']]

# BCH_data_raw = pd.concat(BCH_data_dict)

# %%
BCH_data_WIP = pd.concat(BCH_data_dict).reset_index().drop_duplicates(subset=['Account Number','Date_Time'],  keep='first')

# %% 
# create df with only BCH energy and set datetime index
BCH_kWh = BCH_data_WIP.pivot(index='Date_Time', 
                             columns='Account Number', 
                             values='Net Consumption (kWh)')

#%% calculate hourly energy 

kWh = kW_mod.resample('1h').mean()

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

# %% print table
display(kWh_daily.iloc[-10:,:].map('{:,.2f}'.format))


# %%
# calculate daily average temperature
weather_daily_df = weather_df['Temp (°C)'].resample('1d').mean()
plot_days_df = kWh_daily.join(weather_daily_df)

lines_plot(kWh_daily, plot_days_df, 'MTU and Spyder Total Daily Energy Comparison',
           drop_list=['DHWHP_Spy', 'BCH', 'MTU_BCH_Diff','MTU_Spy_Diff_pct','MTU_BCH_Diff_pct'], 
           legend=True, ylab='Daily Energy (kWh)', 
           start_date=start_date, vs_temp=True)


lines_plot(kWh_daily, plot_days_df, 'MTU and BCH Total Daily Energy Comparison',
           drop_list=['DHWHP_Spy', 'Spy_Sum', 'MTU_Spy_Diff','MTU_Spy_Diff_pct','MTU_BCH_Diff_pct'], 
           legend=True, ylab='Daily Energy (kWh)', 
           start_date=start_date, vs_temp=True)
# %%
# Daily Total Bar Graph

fig, ax = plt.subplots(layout='constrained')

plot_days_df.loc[start_date:,
              ['BCH', 'Main_MTU', 'Spy_Sum']
              ].plot.bar(legend=True, ax=ax)

ax.axhline(color='k')
ax.set_ylabel('Daily Energy Consumption (kWh)')
ax.set_xlabel('Date')
ax.set_xticklabels(ax.get_xticks(), rotation = 90)
ax.set_xticklabels(plot_days_df[start_date:].index.strftime('%Y-%m-%d'))

# for some reason I can't get the line graph to overlay on top of the bar plot
# plt.xticks(rotation = 90)
# plot_days_df.loc[start_date:,'Temp (°C)'
#               ].plot(legend=True, ax=ax)

    

# %% Plot Hourly Energy Total Comparison

# kWh_tot_compare_C = kWh_tot_compare.join(weather_df['Temp (°C)'])

lines_plot(kWh_tot_compare, weather_df.loc[:,['Temp (°C)','Rel Hum (%)']].resample('1d').mean(),
           'MTU and Spyder Total Energy Comparison',
           drop_list=['DHWHP_Spy', 'BCH', 'MTU_BCH_Diff'], legend=True, ylab='Hourly Energy (kWh)', 
           start_date=start_date, vs_temp=True)

lines_plot(kWh_tot_compare,weather_df.loc[:,['Temp (°C)','Rel Hum (%)']].resample('1d').mean(),
           'MTU and BCH Total Energy Comparison',
           drop_list=['DHWHP_Spy', 'Spy_Sum','MTU_Spy_Diff'], legend=True, ylab='Hourly Energy (kWh)', 
           start_date=start_date, vs_temp=True)



# %% Plot Cleaned Power Data Area All

area_plot(kW_mod, 'Power Data Mod - Area', main_line=True,
          drop_list=['DHW_MTU','Main_MTU','Test_MTU'], 
          legend=True, ylab='Power (kW)', start_date=start_date)

# %% Plot Power Total Comparison
if False:
    lines_plot(kW_tot_compare, 'MTU and Spyder Total Power Comparison', 
            drop_list=[], legend=True, ylab='Power (kW)')


# %% Plot Original Data Area Graph
if False:
    area_plot(kW, 'Original Power Data - Area', main_line=True,
            drop_list=['DHW_MTU','Main_MTU','Test_MTU'], 
            legend=True, ylab='Power (kW)')


# %% Plot Hourly Energy Area all channels

area_plot(kWh, 'Hourly Energy - Area', main_line=True,
          drop_list=['DHW_MTU','Main_MTU','Test_MTU'], 
          legend=True, ylab='Hourly Energy (kWh)', start_date=start_date)


# %% Bed Rooms

# lines_plot(kW.loc['2023-10-10 10:00':'2023-10-28 13:20',
#                   ['Heat_Beds','Test_MTU']], 
#                   'Heat_Beds kW Oct 10 to 28', 
#                   start_date='2023-10-10', ylab='kW',
#                   legend=True)

# lines_plot(kW_mod.loc['2023-10-10 10:00':'2023-10-28 13:20',
#                       ['Heat_Beds','Test_MTU']], 
#                       'Heat_Beds kW_mod Oct 10 to 28', 
#                       start_date='2023-10-10', ylab='kW',
#                       legend=True)

# %% Living Room

# lines_plot(kW.loc['2023-10-28 13:30':,['Heat_LvRm','Test_MTU']], 
#            'Heat Living Room kW Oct 28', ylab='kW',
#            start_date='2023-10-28', legend=True)

# lines_plot(kW_mod.loc['2023-10-28 13:30':,['Heat_LvRm','Test_MTU']], 
#            'Heat Living Room kW_mod Oct 28', ylab='kW',
#            start_date='2023-10-28', legend=True)

# %% DHW Spyder vs Test MTU kW

# corrected the DHW MTU polarity at 2024-03-12 8:04 

lines_plot(kW_mod.loc[:,['DHWHP_Spy','Test_MTU']], weather_df, 
           'DHW spy vs test MTU - Power', ylab='kW',
           start_date='2024-03-12', legend=True, vs_temp=True)

# %% DHW Spyder vs Test MTU kWh

lines_plot(kWh.loc[:,['DHWHP_Spy','Test_MTU']].resample('1d').sum(), 
           weather_df.loc[:,['Temp (°C)','Rel Hum (%)']].resample('1d').mean(), 
           'DHW spy vs test MTU - daily energy', ylab='kWh',
           start_date='2024-03-12', legend=True, vs_temp=True)

# %% Use Test MTU for Freezer for the week it was on that circuit

# kW_mod.loc['2023-09-30 12:00':'2023-10-10 09:50','Freezer'] = kW_mod.loc[
#     '2023-09-30 12:00':'2023-10-10 09:50','Test_MTU']

# %% One-off lines Plot

# lines_plot(kW_mod[['Gar_Dryer','Test_MTU']].resample('1H').mean(), 'Dryer', ylab='hourly energy kWh',
#           legend=True)


# lines_plot(kW.filter(like='DishW').resample('1H').mean(), 'Dishwasher', ylab='hourly energy kWh',
#           legend=True)

# lines_plot(kW[['Freezer','Test_MTU']], 'Freezer original power', ylab='kW',
#           legend=True)

# lines_plot(kW, 'original spyder data - lines', 
#            drop_list=['Main_MTU', 'DHW_MTU','Test_MTU'], 
#            legend=True, ylab='Power (kW)')

# %% One-off area Plot

# area_plot(kW_mod.filter(like='Oven'), 'Oven', ylab='power kW',
#           legend=True)

# %% One-off dots Plot

# dots_plot(kW_mod[['Living_Rm','Test_MTU']], 'Living Room Mod', ylab='power kW',
#           legend=True)

# dots_plot(kW_mod[['Bed_Main','Test_MTU']], 'Bed Room Mod', ylab='power kW',
#           legend=True)

# dots_plot(kW_mod[['K_DishW','Test_MTU']], 'Dishwasher', ylab='power kW',
#           legend=True)

# dots_plot(kW_mod[['Freezer','Test_MTU']], 'Freezer Mod', ylab='power kW',
#           legend=True)