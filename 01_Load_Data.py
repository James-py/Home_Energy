#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 05:36:44 2023

@author: james
"""


# %% Import Libraries

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime


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

