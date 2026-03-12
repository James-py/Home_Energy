"""
Created on Thu Aug 24 05:36:44 2023
@author: james

magic command for plots:
%matplotlib

terminal command to get data from Raspberry Pi
rsync -a james@10.0.0.58:/home/james/Data/ ~/repo/Home_Energy/Data_Raw

Use wget in Terminal to get Environment Canada Weather Data
#For multiple years
for year in `seq 2024 2025`;do for month in `seq 1 12`;do wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=52941&Year=${year}&Month=${month}&Day=14&timeframe=1&submit= Download+Data" ;done;done
#For one year
for month in `seq 1 12`;do wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=52941&Year=2025&Month=${month}&Day=14&timeframe=1&submit= Download+Data" ;done
"""

# %% Import Libraries

from IPython.core import display
from matplotlib.colors import ListedColormap
from mpl_axes_aligner import align
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# %% get today's date

today = datetime.datetime.now().date()
yesterday = datetime.datetime.now().date() - datetime.timedelta(days=1)
one_week_ago = today - datetime.timedelta(weeks=1)
four_weeks_ago = today - datetime.timedelta(weeks=4)
Apr_1 = datetime.date(2024, 4, 1)  # datetime.date(2023,10,9)
Dec_5 = datetime.date(2024, 12, 5)

# %% create my custom colourmap

"""
XKCD_COLORS
CSS4_COLORS
tab20
"""

cmap = ListedColormap(
    [
        "darkorange",
        "forestgreen",
        "slategrey",
        "gold",
        "lime",
        "royalblue",
        "lightcoral",
        "lightgreen",
        "blue",
        "red",
        "yellow",
        "seagreen",
        "fuchsia",
        "cyan",
        "indigo",
        "olive",
        "tan",
        "skyblue",
        "salmon",
        "darkseagreen",
        "limegreen",
        "violet",
        "darkblue",
        "chocolate",
        "silver",
        "orange",
        "deeppink",
    ]
)

cmap2 = ListedColormap(["limegreen", "violet", "darkblue", "chocolate", "silver", "orange", "deeppink"])


# %% close all figures

plt.close(fig="all")


# %% update plotting data formats

plt.rcParams["date.autoformatter.hour"] = "%Y-%m-%d %H:%M"
plt.rcParams["date.autoformatter.minute"] = "%Y-%m-%d %H:%M"
plt.rcParams["date.autoformatter.day"] = "%Y-%m-%d"

# %% Turn off plotting of graphs that slow things down by setting to False
allplots = False
otherplots = False

# %% Plotting Functions


def dots_plot(df, caption, vertLine=1000, xlab="Date / Time", ylab="", legend=False, start_date=one_week_ago):
    fig, ax = plt.subplots(1, 1)
    df.loc[start_date:, :].plot(
        ax=ax, colormap=cmap, marker=".", alpha=0.35, linestyle="None", legend=legend, x_compat=True
    )
    if vertLine != 1000:
        plt.axvline(vertLine)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(caption)
    plt.show()


def area_plot(
    df,
    df2,
    caption,
    drop_list=[],
    main_line=False,
    legend=False,
    ylab="",
    start_date=Apr_1,
    end_date=Apr_1 + datetime.timedelta(weeks=2),
    vs_temp=False,
    col2="Temp (°C)",
):
    fig, ax = plt.subplots(1, 1)
    df.loc[start_date:end_date, :].drop(columns=drop_list).plot.area(ax=ax, colormap=cmap, x_compat=True, legend=legend)
    if main_line:
        df.loc[start_date:end_date, :].Main_MTU.plot(color="k", linestyle="-", ax=ax)
        ax.legend()
    ax.set_ylabel(ylab)
    ax.set_xlabel("Date / Time")
    ax.set_title(caption)
    if vs_temp:
        ax2 = ax.twinx()
        colour2 = "tab:blue"
        ax2.plot(df2.loc[start_date:end_date, col2], color=colour2, linestyle="--")
        ax2.set_ylabel("Temperature (°C)", color=colour2)
        ax2.tick_params(axis="y", labelcolor=colour2)
        align.yaxes(ax, 0, ax2, 0, 0.2)
    plt.show()


def lines_plot(
    df,
    df2,
    caption,
    drop_list=[],
    legend=True,
    ylab="",
    start_date=Apr_1,
    end_date=Apr_1 + datetime.timedelta(weeks=2),
    vs_temp=False,
    col2="Temp (°C)",
):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    df.loc[start_date:end_date, :].drop(columns=drop_list).plot(
        alpha=0.75, ax=ax, colormap=cmap, legend=legend, x_compat=True
    )
    ax.set_ylabel(ylab)
    ax.set_xlabel("Date / Time")
    ax.set_title(caption)
    ax.axhline(color="k")
    # y1_max = ax.get_ybound()[1]
    # nticks = 11
    if vs_temp:
        ax2 = ax.twinx()
        colour2 = "tab:blue"
        ax2.plot(df2.loc[start_date:end_date, col2], color=colour2, linestyle="--")
        ax2.set_ylabel("Temperature (°C)", color=colour2)
        ax2.tick_params(axis="y", labelcolor=colour2)
        align.yaxes(ax, 0, ax2, 0, 0.2)
    plt.show()


# plt.close('all')
# lines_plot(kWh_tot_compare,weather_df, 'MTU and BCH Total Energy Comparison',
#            drop_list=['DHWHP_Spy', 'Spy_Sum','MTU_Spy_Diff'], legend=True, ylab='Hourly Energy (kWh)',
#            start_date=start_date, vs_temp=True)


# %% set up folder paths

# root folder for this project path object
CWD = Path(__file__).parent.resolve()
print("\nCWD Folder")
print(CWD)

# recent data folder path object
TED_Data_Path = CWD.joinpath("Data_2024")
print("\nSource Data (Recent) Folder")
print(TED_Data_Path)

# BC Hydro data folder path object
BCH_Path = CWD.joinpath("Data_BC_Hydro")
print("\nBC Hydro Data Folder")
print(BCH_Path)

# Weather data folder path object
Weather_Path = CWD.joinpath("Data_Weather")
print("\nWeather data Folder")
print(Weather_Path)


# %% get file paths to all the csvs
def get_filepaths(f_path, f_pattern):
    file_dict = {}
    for f in f_path.glob(f_pattern):
        file_dict[f.stem] = f
    return file_dict


TED_files = get_filepaths(TED_Data_Path, "*.csv")
Weather_files = get_filepaths(Weather_Path, "*.csv")

# %%
# read Data


def read_data(fp_dict, use_TED_cols=False):
    col_names = [
        "Circuit",
        "Date_Time",
        "kW",
        "cost",
        "Voltage",
        "PF",
    ]
    data_dict = {}
    error_data_dict = {}
    for n, f in fp_dict.items():
        if use_TED_cols:
            try:
                data_dict[n] = pd.read_csv(f, names=col_names)
            except:
                print(f"Error reading {f}.")
                error_data_dict[n] = pd.read_csv(f)
        else:
            data_dict[n] = pd.read_csv(f)
    return pd.concat(data_dict)


weather_data_df = read_data(Weather_files)

# %%
TED_data_df = read_data(TED_files, use_TED_cols=True)

# %%

weather_df = weather_data_df.set_index("Date/Time (LST)")
weather_df.index = pd.to_datetime(weather_df.index)
weather_df.sort_index(inplace=True)
# %%  drop duplicates

data_WIP = TED_data_df.drop_duplicates(subset=["Circuit", "Date_Time"], keep="last")

# %% Pivot and set datetime index
kW = data_WIP.pivot(index="Date_Time", columns="Circuit", values="kW")
kW.index = pd.to_datetime(kW.index)
kW.sort_index(inplace=True)

# %% Data Cleaning - filter noise

for col in kW.columns:
    kW.loc[kW[col] < 0, col] = np.nan
    kW.loc[kW[col] > 12, col] = np.nan

# %% Make working copy of the kW data for data cleaning

kW_mod = pd.DataFrame()

# %%
# Combine double pole dryer data

kW_mod["Gar_Dryer"] = kW.loc[:, ["Gar_Dry1", "Gar_Dry2"]].sum(axis=1) * 1.04

# %% Data Cleaning - Spyder Leg Calibration

calibration_dict = {
    "Bed_G_Off": 0.6,
    "Bed_Main": 0.60,
    "DHWHP_Spy": 0.7,  # sensitive to OAT
    "Freezer": 0.85,
    "Garage": 0.8,
    "Heat_Beds": 1.0,
    "Heat_LvRm": 1.0,
    "K_DishW": 0.55,
    "K_Fridge": 1.0,  # 0.85,
    "K_Oven": 1.1,
    "K_Plg_2MW": 0.75,
    "K_Plg_4Ts": 1.04,
    "K_Plg_5LS": 1.0,
    "K_Plg_IsL": 1.0,
    "K_Plg_IsU": 1.0,
    "Living_Rm": 0.6,
    "Out_Plugs": 0.7,
    "Main_MTU": 1.0,
    "Test_MTU": 1.0,
}

# %%
# Copy desired data from raw and apply the calibration multiplier
for circuit, multiplier in calibration_dict.items():
    kW_mod[circuit] = kW.filter(like=circuit) * multiplier

kW_mod.sort_index(axis=1, inplace=True)

# %% Fix Heating V2


def fix_heating_v2(df_mod, df, circuit, dc_offset, linear_scale, cutoff):
    df_mod.loc[:, circuit] = df.loc[:, circuit]

    # shift everything down by dc_offset
    df_mod.loc[:, circuit] = df_mod.loc[:, circuit] - dc_offset
    # but don't make negative values
    df_mod.loc[df_mod[circuit] < 0, circuit] = 0

    # Linearly scale the signal
    df_mod.loc[:, circuit] = df_mod.loc[:, circuit] * linear_scale

    # cut off spikes above the cutoff value
    df_mod.loc[df_mod[circuit] > cutoff, circuit] = cutoff


fix_heating_v2(kW_mod, kW, "Heat_LvRm", 0.15, 1.2, 2.6)

fix_heating_v2(kW_mod, kW, "Heat_Beds", 0.15, 1.4, 3.0)


# %% Fill Missing Power Load
# spyder CTs are less accurate and don't reliably detect loads <100 W
# this fills in a minimum load when ct dectects less than the minimum


def fill_one_circuit(df_mod, df, circuit, load):
    df_mod.loc[df[circuit] < load, circuit] = load


# DHW_HP: 10 W appears to be a standby load
fill_one_circuit(kW_mod, kW, "DHWHP_Spy", 0.01)
# Living Room: 30 W appears to be the router that is always on
fill_one_circuit(kW_mod, kW, "Living_Rm", 0.03)
# various standby and chargers in bedrooms are still a small load
fill_one_circuit(kW_mod, kW, "Bed_Main", 0.01)
# raspberry pi and laptop dock
fill_one_circuit(kW_mod, kW, "Bed_G_Off", 0.01)
# When freezer is on, the spyder seems to underestimate the load sometimes
kW_mod.loc[kW.Freezer.between(0.01, 0.03), "Freezer"] = 0.05

# %% Total Power Comparison Calcs
# - uses spyder data for DHW
# fill Main MTU data using spyder sum

kW_tot_compare = pd.DataFrame(kW_mod["Main_MTU"])
kW_tot_compare["Spy_Sum"] = kW_mod.drop(columns=["Main_MTU", "Test_MTU"]).sum(axis=1)
kW_tot_compare["Main_MTU"] = kW_tot_compare["Main_MTU"].fillna(kW_tot_compare.Spy_Sum)

# %% import BC Hydro data

BCH_data_files = {}
for f in BCH_Path.glob("*consumption*.csv"):
    BCH_data_files[f.stem] = f

BCH_data_dict = {}
for n, f in BCH_data_files.items():
    BCH_data_dict[n] = pd.read_csv(f)

for n, df in BCH_data_dict.items():
    df["Date_Time"] = pd.to_datetime(df["Interval Start Date/Time"], yearfirst=True).dt.floor("Min")
    if df["Account Number"].dtype == "O":
        df["Account Number"] = pd.to_numeric(df["Account Number"].str.removeprefix("'"))

# %%
# Combine BCH data into one dataframe and drop duplicates

BCH_data_WIP = (
    pd.concat(BCH_data_dict).reset_index().drop_duplicates(subset=["Account Number", "Date_Time"], keep="first")
)

# %%
# create df with only BCH energy and set datetime index
BCH_kWh = BCH_data_WIP.pivot(index="Date_Time", columns="Account Number", values="Net Consumption (kWh)").sum(axis=1)

# %% calculate hourly energy

kWh = kW_mod.resample("1h").mean()

# %% Total Energy Compare and combine TED and BCH data

kWh_tot_compare = pd.DataFrame(kWh[["Main_MTU", "DHWHP_Spy"]])
kWh_tot_compare["Spy_Sum"] = kWh.drop(columns=["Main_MTU", "Test_MTU"]).sum(axis=1)
kWh_tot_compare["Main_MTU"] = kWh["Main_MTU"].fillna(kWh_tot_compare.Spy_Sum)
kWh_tot_compare["MTU_Spy_Diff"] = kWh_tot_compare.Main_MTU - kWh_tot_compare.Spy_Sum

# Comment this out until I get hourly data
# kWh_tot_compare.loc[kWh_tot_compare['MTU_Spy_Diff']<0, 'MTU_Spy_Diff'] = np.nan
# kWh_tot_compare["BCH"] = BCH_kWh
# kWh_tot_compare["MTU_BCH_Diff"] = kWh_tot_compare.Main_MTU - kWh_tot_compare.BCH

# %% Daily Energy Totals

kWh_daily = kWh_tot_compare.resample("1d").sum()
# work-around because I only have daily BCH data right now
kWh_daily["BCH"] = BCH_kWh
kWh_daily["MTU_BCH_Diff"] = kWh_daily.Main_MTU - kWh_daily.BCH
kWh_daily["MTU_Spy_Diff_pct"] = kWh_daily.MTU_Spy_Diff / kWh_daily.Spy_Sum
kWh_daily["MTU_BCH_Diff_pct"] = kWh_daily.MTU_BCH_Diff / kWh_daily.BCH

# print table
kWh_daily.iloc[-10:, :].map("{:,.2f}".format)

# %% DHW Daily Average
ave_daily_DHW = kWh_daily.DHWHP_Spy.mean()
dollar_per_kWh = 0.11
print(
    "Average daily DHW HP Energy Consumption:",
    round(ave_daily_DHW, 2),
    f"\n {round(ave_daily_DHW, 2)} kWh",
    f"\n ${round(ave_daily_DHW * dollar_per_kWh, 2)}",
    "\n",
)

print(
    "Extrapolated Annual DHW HP Energy Consumption:",
    round(ave_daily_DHW, 2),
    f"\n {round(ave_daily_DHW * 365, 2)} kWh",
    f"\n ${round(ave_daily_DHW * dollar_per_kWh * 365, 2)}",
    "\n",
)


# %% DHW total consumption from Apr 1 to Dec 5

DHW_kWh_249_days = kWh_daily.DHWHP_Spy["2024-04-01":"2024-12-05"].sum()
dollar_per_kWh = 0.11
print(
    "Total DHW HP Energy Consumption from Apr 1, 2024 to Dec 5, 2024 (249 days):",
    f"\n {round(DHW_kWh_249_days, 2)} kWh",
    f"\n ${round(DHW_kWh_249_days * dollar_per_kWh, 2)}",
)


# %%
# calculate daily average temperature
weather_daily_df = weather_df["Temp (°C)"].resample("1d").mean()
plot_days_df = kWh_daily.join(weather_daily_df)

lines_plot(
    kWh_daily,
    plot_days_df,
    "MTU and Spyder Total Daily Energy Comparison",
    drop_list=["DHWHP_Spy", "BCH", "MTU_BCH_Diff", "MTU_Spy_Diff_pct", "MTU_BCH_Diff_pct"],
    legend=True,
    ylab="Daily Energy (kWh)",
    start_date=start_date,
    vs_temp=True,
)


lines_plot(
    kWh_daily,
    plot_days_df,
    "MTU and BCH Total Daily Energy Comparison",
    drop_list=["DHWHP_Spy", "Spy_Sum", "MTU_Spy_Diff", "MTU_Spy_Diff_pct", "MTU_BCH_Diff_pct"],
    legend=True,
    ylab="Daily Energy (kWh)",
    start_date=start_date,
    vs_temp=True,
)
# %%
# Daily Total Bar Graph

fig, ax = plt.subplots(layout="constrained")

plot_days_df.loc[start_date:, ["BCH", "Main_MTU", "Spy_Sum"]].plot.bar(legend=True, ax=ax)

ax.axhline(color="k")
ax.set_ylabel("Daily Energy Consumption (kWh)")
ax.set_xlabel("Date")
ax.set_xticklabels(ax.get_xticks(), rotation=90)
ax.set_xticklabels(plot_days_df[start_date:].index.strftime("%Y-%m-%d"))


# %%
# Plot Cleaned Power Data Area All
area_plot(
    kW_mod,
    weather_df,
    "Power Data Mod - Area",
    main_line=False,
    drop_list=["Main_MTU", "Test_MTU"],
    legend=True,
    ylab="Power (kW)",
)


# %%
# change matplotlib backend
plt.switch_backend("tkagg")
# plt.switch_backend("ipympl")
# %matplotlib


# %% Plot Hourly Energy Area all channels

area_plot(
    kWh,
    weather_df,
    "Hourly Energy - Area",
    main_line=True,
    drop_list=["Main_MTU", "Test_MTU"],
    legend=True,
    ylab="Hourly Energy (kWh)",
    start_date=start_date,
    end_date=end_date,
    vs_temp=False,
)


# plt.close(fig='all')

# %%
