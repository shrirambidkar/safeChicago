# -*- coding: utf-8 -*-
#**********************************************************************
# FILENAME :        TrendAnalysis V1.0.py            
#
# DESCRIPTION :
#       Provide indepth trend analysis by providing different graphs for better visualization
#       of the data for further analysis
#
# NOTES :
#       Provide trend analysis using different graph plotting techniques
# 
# AUTHOR :    Shriram Bidkar        START DATE :    Nov 20 2019
#
# CHANGES :
# VERSION   DATE            WHO                 DETAIL
# V1.0      11/20/2019      Shriram Bidkar      First baseline version V1.0
#
#**********************************************************************

#prepare
# Data manipulation modules
import pandas as pd        # data manipulation package
import numpy as np         # n-dimensional arrays

# For plotting
import matplotlib.pyplot as plt      # For base plotting
# Seaborn is a library for making statistical graphics
# in Python. It is built on top of matplotlib and 
#  numpy and pandas data structures.
import seaborn as sns                # Easier plotting

# Misc
import os
import sys
from datetime import datetime

# Show graphs in a separate window
#%matplotlib qt5

# Import Dataset
# Read data file and make the current path as absolute path 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#data = pd.read_csv("/Users/shrirambidkar/Documents/Education/CS 235 - Data Mining Techniques/Project/data/gvaData.csv")
data = pd.read_csv("gvaData.csv")
data = data[(data.state == 'Illinois') & data["city_or_county"].str.contains("Chicago")]  # filter the data based on Illinois state
os.chdir(dname)

# data wrangling
data['date'] = pd.to_datetime(data['date'])      # Convert to datetime
data.insert(loc=2, column='date_year', value=data['date'].dt.year) # split the date to have year with new colume date_year
data.insert(loc=3, column='date_month', value=data['date'].dt.month) # split the date to have month with new colume date_month
data.insert(loc=4, column='date_day', value=data['date'].dt.day) # split the date to have day with new colume date_day
data.insert(loc=5, column='date_wday', value=data['date'].dt.weekday_name) # split the date to have which day of the week with new colume date_wday
data.insert(loc=6, column='date_wend', value=np.where(((data['date_wday'] == 'Saturday') | (data['date_wday'] == 'Sunday')) , 'Weekends', 'Weekdays')) # split the day of week to weekend and weekdays with new colume date_wend
data.insert(loc=22, column='participant_male', value=data['participant_gender'].str.count('Male')) # split participant_gender column to have participant_male column
data.insert(loc=23, column='participant_female', value=data['participant_gender'].str.count('Female')) # split participant_gender column to have participant_female column
data.insert(loc=24, column='participant_injured', value=data['participant_status'].str.count('Injured')) # split participant_status column to have participant_injured column
data.insert(loc=25, column='participant_arrested', value=data['participant_status'].str.count('Arrested')) # split participant_status column to have participant_arrested column
data.insert(loc=26, column='participant_killed', value=data['participant_status'].str.count('Killed')) # split participant_status column to have participant_killed column
data.insert(loc=27, column='participant_unharmed', value=data['participant_status'].str.count('Unharmed')) # split participant_status column to have participant_unharmed column
data.insert(loc=28, column='participant_victim', value=data['participant_type'].str.count('Victim')) # split participant_type column to have victim column
data.insert(loc=29, column='participant_subject_suspect', value=data['participant_type'].str.count('Subject-Suspect')) # split participant_type column to have subject-suspect column
data.insert(loc=30, column='participant_child', value=data['participant_age_group'].str.count('Adult')) # split participant_age_group column to have adult column
data.insert(loc=31, column='participant_teen', value=data['participant_age_group'].str.count('Teen')) # split participant_age_group column to have Teen column
data.insert(loc=32, column='participant_adult', value=data['participant_age_group'].str.count('Child'))  # split participant_age_group column to have Child column

# Drop redundent columns which are not required
data = data.drop(columns=['date', 'incident_id', 'incident_url', 'source_url', 'incident_url_fields_missing', 'incident_characteristics', 'location_description', 'sources', 'notes', 'participant_name', 'participant_relationship', 'participant_gender', 'participant_status', 'participant_type','participant_age_group'] ) 

# Plot Graphs
# Plot Joint Distribution graph which illustrates year wise number of participants killed 

g = sns.jointplot(data['date_year'], data['participant_killed'], data, kind='reg', dropna =True)
print('Please close the graph of Year Vs # of Killed which is in a popup window after review.')
plt.show()

# Plot Histograms of congressional district where crime rate has been high
data['congressional_district'].fillna(0, inplace=True) # update all empty values (NaN) to 0
g = sns.distplot(data['congressional_district'], hist=True, kde=True, bins=range(1,53)); # remove all zero values from the plot
print('Please close the graph of congressional district which is in a popup window after review.')
plt.show()

# Plot Kernel Density graph to illustrate killed Vs injured 
sns.kdeplot(data['n_killed'], label='Killed', shade=True)
sns.kdeplot(data['n_injured'], label='Injured', shade=True)
plt.xlabel('Impacted');
print('Please close the graph of # of participants killed vs injured which is in a popup window after review.')
plt.show()

# Plot  Violin graph to illustrate male Vs female
sns.violinplot("participant_male", "participant_female", data=data );     # x-axis has categorical variable
sns.violinplot( "participant_female", "participant_male", data=data );    # y-axis has categorical variable
print('Please close the graph of male and female participants injured in gun violance which is in a popup window after review.')
plt.show()

# Plot  Violin graph to illustrate child victims and their distributions over weekdays and weekends
sns.violinplot("participant_child", "participant_victim",
               hue="date_wend",
               data=data,
               split=True,         
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
print('Please close the graph of gun violance victim where child was involved which is in a popup window after review.')
plt.show()

# build a Box plots to identify patterns od teens involved over the week days
sns.boxplot("date_wday", "participant_teen", data= data)
print('Please close the graph of week day when teen was involved which is in a popup window after review.')
plt.show()

g = sns.FacetGrid(data, hue="date_wend", palette="Set1", size=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "participant_male", "n_killed", s=50, alpha=.7, linewidth=.5, edgecolor="white")
g.add_legend();
print('Please close the graph of male killed in gun violance which is in a popup window after review.')
plt.show()

# End of the file