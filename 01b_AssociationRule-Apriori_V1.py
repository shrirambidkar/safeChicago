# -*- coding: utf-8 -*-
#**********************************************************************
# FILENAME :        AssociationRule-Apriori V1.py             
#
# DESCRIPTION :
#       Provide trend analysis using Association Rules technique (Apriori) of data mining 
#       which will provide correlation of one feature / attribute with another
#
# NOTES :
#       Provide trend analysis using Association Rules
#
# AUTHOR :    Shriram Bidkar        START DATE :    Nov 19 2019
#
# CHANGES :
# VERSION   DATE            WHO                 DETAIL
# V1.0      11/19/2019      Shriram Bidkar      First baseline version V1.0
#
#**********************************************************************

#prepare
# Data manipulation modules
import pandas as pd        # data manipulation package
import numpy as np         # n-dimensional arrays
from datetime import datetime

# For plotting
import matplotlib.pyplot as plt      # For base plotting
# Seaborn is a library for making statistical graphics
# in Python. It is built on top of matplotlib and 
#  numpy and pandas data structures.
import seaborn as sns                # Easier plotting

# For Association Rules Technique  
from apyori import apriori           # Apriori Algorithm

# Misc
import os
import sys

# Import Dataset
# Read data file and make the current path as absolute path 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
data = pd.read_csv("gvaData.csv")
data = data[(data.state == 'Illinois') & data["city_or_county"].str.contains("Chicago")]  # filter the data based on Illinois state
os.chdir(dname)

# Data Preprocessing (Cleaning; Reduction; Transformation)
data['date'] = pd.to_datetime(data['date']).dt.weekday_name      # Convert to datetime
data.insert(loc=1, column='gender_male', value=np.where((data['participant_gender'].str.count('Male') > 0), 'Male','')) # split participant_gender column to have gender_male binary column
data.insert(loc=2, column='gender_female', value=np.where((data['participant_gender'].str.count('Female') > 0), 'Female','')) # split participant_gender column to have gender_female binary column
data.insert(loc=3, column='age_adult', value=np.where((data['participant_age_group'].str.count('Adult') > 0), 'Adult','')) # split participant_age_group column to have binary age_adult column
data.insert(loc=4, column='age_teen', value=np.where((data['participant_age_group'].str.count('Teen') > 0), 'Teen','')) # split participant_age_group column to have binary age_teen column
data.insert(loc=5, column='age_child', value=np.where((data['participant_age_group'].str.count('Child') > 0), 'Child','')) # split participant_age_group column to have binary age_child column

# Eliminate null values by verifying the distribution and append the attribute / feature appropriately

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='congressional_district') # check data distribution for this column 
data['congressional_district'].fillna(data['congressional_district'].median(), inplace=True) # since data is skewed we use median
data['congressional_district'] = 'Congressional District - ' + data['congressional_district'].astype(str) # append a string to the data which will help us understand the assicoation better

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='state_house_district') # check data distribution for this column 
data['state_house_district'].fillna(data['state_house_district'].median(), inplace=True) # since data is skewed we use median
data['state_house_district'] = 'State House District - ' + data['state_house_district'].astype(str) # append a string to the data which will help us understand the assicoation better

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='state_senate_district') # check data distribution for this column 
data['state_senate_district'].fillna(data['state_senate_district'].median(), inplace=True) # since data is skewed we use median
data['state_senate_district'] = 'State Senate District - ' + data['state_senate_district'].astype(str) # append a string to the data which will help us understand the assicoation better

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='latitude') # check data distribution for this column 
data['latitude'].fillna(data['latitude'].median(), inplace=True) # since data is skewed we use median

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='longitude') # check data distribution for this column 
data['longitude'].fillna(data['longitude'].median(), inplace=True) # since data is skewed we use median

# Combine latitude and longitude so that it is viewed together instead of treating it seperate 
data.insert(loc=6, column='location', value=(data['latitude'].round(3).astype(str) + ', ' + data['longitude'].round(3).astype(str))) 

# verify the target column for missing values and remove them 
data['participant_status'].isna().sum() # check if there are any null values for target
data.dropna(how='any', subset=['participant_status'], inplace=True) # remove null data as it may be difficult to fill them
data['participant_status'] = np.where(
                                    ((data['participant_status'].str.count('Killed') > 0 ) |    # if the status is either killed or injured we mark
                                     (data['participant_status'].str.count('Injured') > 0)      # the column as KilledORInjured 
                                    ) , 'KilledORInjured', 'Unharmed'                           # else we mark it as unharmed
                                     )  

data.groupby('participant_status').size() # verify dimensions of column

# Columns removed from data set as it was not adding any value to the dataset
data = data.drop(columns=[
    'incident_id',                      # ID is not required for analysis
    'incident_url',                     # URL is similar to ID and is not required for analysis
    'source_url',                       # URL is similar to ID and is not required for analysis
    'incident_url_fields_missing',      # Not required for analysis
    'incident_characteristics',         # Since characterstics are after the incident may not be useful for analysis
    'location_description',             # Getting captured as part of latitude and longitude 
    'sources',                          # Since sources are after the incident may not be useful for analysis
    'notes',                            # Since notes are after the incident may not be useful for analysis
    'participant_name',                 # Name is similar to ID and is not required for analysis
    'participant_relationship',         # Since relationship are determined after the incident may not be useful for analysis
    'participant_gender',               # Already captured under gender_male and gender_female columns
    'participant_type',                 # Since participant type is captured as age category columns, this column may not be required
    'participant_age_group',            # Since age groupis captured as age category columns, this column may not be required
    'gun_stolen',                       # Since guns stolen is identified  after the incident may not be useful for analysis
    'participant_age',                  # Since age is captured as age category columns, this column may not be required
    'gun_type',                         # Since guns type is identified  after the incident may not be useful for analysis
    'state',                            # capturing only Illinois state data so we can remove this column / attribute 
    'address',                          # Capturing lat and long so this field is not required
    'n_guns_involved',                  # Guns involved is identified  after the incident may not be useful for analysis
    'latitude',                         # This column is combined with location column 
    'longitude',                        # This column is combined with location column 
    'n_killed',                         # number of killed and injured is identified after the incident, may not be useful for analysis
    'n_injured'                         # similar to above comment, may not be useful for analysis
    ] ) # Drop redundent columns which are not required

data.columns # verify final list of features which will be used to identify assocaition

# We will be using Apriori package and for which we need list of list and it does not require header for the data
# We will remove the header and create a list from dataset 
new_header = data.loc[16] # identify the data within dataset which does not have null values in the record whcih will be treated as header 
apriori_data = data[1:] # get a get from dataset
apriori_data.columns = new_header # copy the new header to old dataset
apriori_data.shape # verify dimensions of the dataset

apriori_list = apriori_data.values.tolist() # convert it to list

# Utilize Apriori package which takes 4 parameters namely -
#  1.   min_support which denotes default popularity of an item and in this case we are assuming most of the crimes happens 
#       during friday, saturday and sunday for last 5 years of data and hence we are calculating = ((52*5)*3)/11639 = 0.0670
#       where 52 is weeks, 5 is years and 3 is days (Friday, Saturday and Sunday)
#  2.   min_confidence denotes likelihood that one feature (gender) is influcing another one feature (killedORinjured). 
#       In this case we are assuming we have high likely that we have 50% chance that would happen hence 0.5
#  3.   min_lift denotes increase in the ratio of one parameter/feature on another parameter / feature. 
#       In this case it is we are taking it as 4
#  4.   Last parameter is min_length refers to number of features we want to include in the rule and in our case we wanted to be 5 
ar = apriori(apriori_list, min_support=0.0670, min_confidence=0.5, min_lift=4, min_length=5)

ar_results = list(ar) # put the results in the list

print('Number of associations identified: ', len(ar_results)) # verify the results
print('Sample first row: ', ar_results[0]) # verify sample of the result 

# print all the results in formatted way which provides meaning ful insight on Confidence / list 
for item in ar_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + str(items[0]) + " -> " + str(items[1]))

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

# End of the file