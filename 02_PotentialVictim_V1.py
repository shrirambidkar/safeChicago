# -*- coding: utf-8 -*-
#**********************************************************************
# FILENAME :        PotetialVictim V1.0.py             
#
# DESCRIPTION :
#       For this in this file we have used  binary classification methods like decision tree and 
#       SVM (Support Vector Machines) to identify if user of the application is potential victim. 
#       And finally we evaluated performance of the algorithm using confusion matrix and AUC.
#
# NOTES :
#       To notify the user if they are tagged as potential victim with their 
#       given age, gender & geo-location.
#
# AUTHOR :    Shriram Bidkar        START DATE :    Nov 19 2019
#
# CHANGES :
# VERSION   DATE            WHO                 DETAIL
# V1.0      11/19/2019      Shriram Bidkar      First baseline version V1.0
#
#**********************************************************************

# Import important packages  
# Data manipulation modules
import pandas as pd        # data manipulation package
import numpy as np         # n-dimensional arrays

# For plotting
import matplotlib.pyplot as plt      # For base plotting
import seaborn as sns                # Easier plotting

# For preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA

# To split the date into training and test
from sklearn.model_selection import train_test_split

# To build models
from sklearn import tree, svm

# Validate models
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn import metrics

# Misc
import os
import sys
from datetime import datetime

# Import Dataset
# Read data file and make the current path as absolute path 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
data = pd.read_csv("gvaData.csv")
data = data[(data.state == 'Illinois')]  # filter the data based on Illinois state
os.chdir(dname)

# Data Preprocessing (Cleaning; Reduction; Transformation)
data['date'] = pd.to_datetime(data['date'])      # Convert to datetime
data.insert(loc=5, column='monday', value=np.where((data['date'].dt.weekday_name) == 'Monday', 1, 0))  # split the date to seperate column (day of the week - Monday)
data.insert(loc=6, column='tuesday', value=np.where((data['date'].dt.weekday_name) == 'Tuesday', 1, 0)) # split the date to seperate column (day of the week - Tueday)
data.insert(loc=7, column='wednesday', value=np.where((data['date'].dt.weekday_name) == 'Wednesday', 1, 0)) # split the date to seperate column (day of the week - Wednesday)
data.insert(loc=8, column='thursday', value=np.where((data['date'].dt.weekday_name) == 'Thursday', 1, 0)) # split the date to seperate column (day of the week - Thursday)
data.insert(loc=9, column='friday', value=np.where((data['date'].dt.weekday_name) == 'Friday', 1, 0)) # split the date to seperate column (day of the week - Friday)
data.insert(loc=10, column='saturday', value=np.where((data['date'].dt.weekday_name) == 'Saturday', 1, 0)) # split the date to seperate column (day of the week - Saturday)
data.insert(loc=11, column='sunday', value=np.where((data['date'].dt.weekday_name) == 'Sunday', 1, 0)) # split the date to seperate column (day of the week - Sunday)
data.insert(loc=12, column='gender_male', value=np.where((data['participant_gender'].str.count('Male') > 0), 1, 0)) # split participant_gender column to have gender_male binary column
data.insert(loc=13, column='gender_female', value=np.where((data['participant_gender'].str.count('Female') > 0), 1, 0)) # split participant_gender column to have gender_female binary column
data.insert(loc=14, column='age_adult', value=np.where((data['participant_age_group'].str.count('Adult') > 0), 1, 0)) # split participant_age_group column to have binary age_adult column
data.insert(loc=15, column='age_teen', value=np.where((data['participant_age_group'].str.count('Teen') > 0), 1, 0)) # split participant_age_group column to have binary age_teen column
data.insert(loc=16, column='age_child', value=np.where((data['participant_age_group'].str.count('Child') > 0), 1, 0)) # split participant_age_group column to have binary age_child column

# Validate target attribute for its correctness
data['participant_status'].isna().sum() # check if there are any null values for target
data.dropna(how='any', subset=['participant_status'], inplace=True) # remove null data as it may be difficult to fill them
# convert the string field to binary value. If target is either injured or killed we mark them as potential victim
data['participant_status'] = np.where(
                                    ((data['participant_status'].str.count('Killed') > 0 ) | 
                                     (data['participant_status'].str.count('Injured') > 0)
                                    ) , 1, 0
                                     )  
data.rename(columns = {'participant_status':'killedORinjured'}, inplace = True) # rename the column for better understanding

# Columns removed from data set as it was not adding any value to the dataset
data = data.drop(columns=[
    'date',                             # since the date is already converted into weekdays we do not need a specific date
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
    'n_killed',                         # number of killed and injured is identified after the incident, may not be useful for analysis
    'n_injured'                         # similar to above comment, may not be useful for analysis
    ] ) # Drop redundent columns which are not required

# Preprocessing of data - check for missing valies
data.isna().sum() # check which columns have missing data

# check if data is symetrical (measures of central tendency). if data is skewed use median over mean

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='congressional_district') # check data distribution for this column 
data['congressional_district'].fillna(data['congressional_district'].median(), inplace=True) # since data is skewed we use median 

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='state_house_district') # check data distribution for this column 
data['state_house_district'].fillna(data['state_house_district'].median(), inplace=True) # since data is skewed we use median

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='state_senate_district') # check data distribution for this column 
data['state_senate_district'].fillna(data['state_senate_district'].median(), inplace=True) # since data is skewed we use median

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='latitude') # check data distribution for this column 
data['latitude'].fillna(data['latitude'].median(), inplace=True) # since data is skewed we use median

# This below step of plotting Histo has been delebrately commented as this is final delivery. We decided to go with median because when we plotted the data was skewed
#data.hist(column='longitude') # check data distribution for this column 
data['longitude'].fillna(data['longitude'].median(), inplace=True) # since data is skewed we use median

data.isna().sum() # verify if data is populated with filled values

# encode categorical variable / feature (city_or_county) 
data.groupby('city_or_county').size() 

city_encoder = LabelEncoder() # instantiate labelEncoder from scikitlearn
city_labels = city_encoder.fit_transform(data['city_or_county']) # fit and transform the column
city_mappings = {index: label for index, label in 
                  enumerate(city_encoder.classes_)} # convert column city_or_county column from text to numeric values
data['city_or_county'] = city_labels # Add numeric values back to column 

# check values of column to verify other categorical variables and convert them multiple dummy/indicator variables.
data.groupby('state_senate_district').size() # verify column state_senate_district for data distribution across different values of categorical variable
data.groupby('state_house_district').size() # verify column state_house_district for data distribution across different values of categorical variable
data.groupby('congressional_district').size() # verify column congressional_district for data distribution across different values of categorical variable
data.groupby('city_or_county').size() # verify column city_or_county for data distribution across different values of categorical variable


# covert categorical variables into multiple dummy/indicator variables.
ohe_data = pd.get_dummies(data, columns=[
    'state_senate_district', 
    'state_house_district', 
    'congressional_district', 
    'city_or_county'],
    drop_first=True)

ohe_data.columns

# split the dateset chicago_ohe_df into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
                                                ohe_data, 
                                                ohe_data['killedORinjured'], 
                                                test_size=0.3) # split the data into 70/30 with test size of 30%

print X_train.shape, y_train.shape # (148435, 13040) (148435,)
print X_test.shape, y_test.shape # (63616, 13040) (63616,)

# Normalize the data from which will convert all the data from [0-1] scale which will help the model to perform better
X_train_nor = preprocessing.normalize(X_train)
X_test_nor = preprocessing.normalize(X_test)

# Use Principal Component Analysis to reduce the features / attributes
# We will do this in two steps - 
#       1. Identify total components and verify which compoenets contribute most 
#       2. Use the most contributing compoents again through PCA and build the training data

# Since we are packaging this for final deliverable we do not want to validate minimum PCA and hence below few steps has been commented but these 
# steps are necessary to determine what should be the components for PCA.

# Step 1 where we identify total components and verify which compoenets contribute most 
#pca = PCA() 
#X_train_pca = pca.fit_transform(X_train_nor)
#X_test_pca = pca.transform(X_test_nor)

#explained_variance = (pca.explained_variance_ratio_)*100 # check how much components contribute the most

# Step 2 run PCA this time with optimal components
pca = PCA(n_components=11)
X_train_pca = pca.fit_transform(X_train_nor)
X_test_pca = pca.transform(X_test_nor)

# Built model and test it 
# create a function which will plot AUC and ROC once the model is built and tested to determine accuracy of models
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# Using Decision Tree Classifier from scikitLearn package
gunViolanceDTCModel = tree.DecisionTreeClassifier(max_depth=3, random_state=7)
gunViolanceDTCModel.fit(X_train_pca, y_train)
y_predict = gunViolanceDTCModel.predict(X_test_pca)

# Validate accuracy using ROC and AUC for Decision Tree Classifier
gvdtcProbs = gunViolanceDTCModel.predict_proba(X_test_pca)
gvdtcProbs = gvdtcProbs[:, 1]
auc = roc_auc_score(y_test, gvdtcProbs)
print('AUC (Area Under Curve) for Decision Tree : %.2f' % auc) 
fpr, tpr, thresholds = roc_curve(y_test, gvdtcProbs)
print('Please close the ROC graph for decision tree which is in a popup window after review.')
plot_roc_curve(fpr, tpr) # Plot AUC graphs with accuracy
plt.show()

# Using SVM (Simple Vector Machines) Classifier from scikitLearn package
gunViolanceSVMModel = svm.SVC(kernel='linear', probability=True, random_state=27)
gunViolanceSVMModel.fit(X_train_pca, y_train)
y_pred = gunViolanceSVMModel.predict(X_test_pca)

# Validate accuracy using ROC and AUC for SVM
gvsvmProbs = gunViolanceSVMModel.predict_proba(X_test_pca)
gvsvmProbs = gvsvmProbs[:, 1]
auc = roc_auc_score(y_test, gvsvmProbs)
print('AUC (Area Under Curve) for SVM : %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, gvsvmProbs)
print('Please close the ROC graph for SVM which is in a popup window after review.')
plot_roc_curve(fpr, tpr) # Plot AUC graphs with accuracy
plt.show()

# End of the file
