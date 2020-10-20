#!/usr/bin/env python
# coding: utf-8

# In[108]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from acquire import get_zillow_data


# In[109]:


df= get_zillow_data()
df.head()


# In[110]:


# Let's figure out how much data is missing where
def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing / rows
    cols_missing = pd.DataFrame({'number_missing_rows': num_missing, 'percent_rows_missing': pct_missing})
    return cols_missing


# In[111]:


#checking which columns have the most null rows
nulls_by_column = nulls_by_col(df)
nulls_by_column.sort_values(by="percent_rows_missing", ascending=False, inplace=True)
nulls_by_column           


# In[163]:


def clean_zillow(cached=True):
    '''This function acquires and prepares the zillow data from a local csv, default. Passing cached=False acquires fresh data from sql and writes to csv.'''
    # use my aquire function to read data into a df from a csv file
    df = get_zillow_data()
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop duplicate columns and remove columns with more than 50% nulls
    df = df.drop(columns=['unitcnt','propertylandusedesc','heatingorsystemdesc','propertyzoningdesc','heatingorsystemtypeid','taxdelinquencyflag','taxdelinquencyyear','yardbuildingsqft17','finishedsquarefeet50','finishedfloor1squarefeet','fireplacecnt','threequarterbathnbr','pooltypeid7','poolcnt','numberofstories','airconditioningdesc','garagetotalsqft','garagecarcnt','regionidneighborhood','hashottuborspa','pooltypeid2','poolsizesum','pooltypeid10','typeconstructiontypeid','typeconstructiondesc','architecturalstyledesc','finishedsquarefeet6','fireplaceflag','yardbuildingsqft26','finishedsquarefeet13','storytypeid','storydesc','basementsqft','finishedsquarefeet15','buildingclassdesc','architecturalstyletypeid','airconditioningtypeid','buildingclasstypeid','buildingqualitytypeid','decktypeid','architecturalstyletypeid.1','airconditioningtypeid.1','heatingorsystemtypeid.1','propertylandusetypeid.1','buildingclasstypeid.1', 'storytypeid.1', 'typeconstructiontypeid.1','id.1','Unnamed: 0','calculatedbathnbr', 'fips', 'latitude', 'longitude', 'regionidcounty', 'roomcnt', 'yearbuilt', 'assessmentyear', 'propertycountylandusecode', 'propertylandusetypeid', 'parcelid.2','parcelid.1'])
    #removing columns
    df.replace(',','', regex=True, inplace=True)
    #handling nan's
    df=df.fillna(df.mean())
    return df


# In[164]:


df=clean_zillow()


# In[165]:


df.head()


# In[166]:


df.isna().sum()


# In[167]:


#checking null count after making clean data frame
nulls_by_column = nulls_by_col(df)
nulls_by_column.sort_values(by="percent_rows_missing", ascending=False, inplace=True)
nulls_by_column           


# In[147]:


# #making my split, train, test data
train_validate, test = train_test_split(df, test_size=.2, 
                                         random_state=42,
                                           )
train, validate = train_test_split(train_validate, test_size=.3, 
                                  random_state=42,
                                         ) 


# In[148]:


df.info()


# In[125]:


#combining my split, train, test data and my clean data into one dataframe
def prep_zillow_data():
    '''This function will return a data frame holding both my clean data and the split/train/test data.'''
    df = clean_zillow()
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=42)
    X_train = train.drop(columns='logerror')
    X_validate = validate.drop(columns='logerror')
    X_test = test.drop(columns='logerror')

    y_train = train['logerror']
    y_validate = validate['logerror']
    y_test = test['logerror']
    return train, validate, test


# In[129]:


def scaled_data(train, validate, test):

    train = train.drop(['logerror','calculatedfinishedsquarefeet'], axis=1)
    validate = validate.drop(['logerror','calculatedfinishedsquarefeet'], axis=1)
    test = test.drop(['logerror','calculatedfinishedsquarefeet'], axis=1)

    # 1. Create the Scaling Object
    scaler = sklearn.preprocessing.StandardScaler()

    # 2. Fit to the train data only
    scaler.fit(train)

    # 3. use the object on the whole df
    # this returns an array, so we convert to df in the same line
    train_scaled = pd.DataFrame(scaler.transform(train))
    validate_scaled = pd.DataFrame(scaler.transform(validate))
    test_scaled = pd.DataFrame(scaler.transform(test))

    # the result of changing an array to a df resets the index and columns
    # for each train, validate, and test, we change the index and columns back to original values

    # Train
    train_scaled.index = train.index
    train_scaled.columns = train.columns

    # Validate
    validate_scaled.index = validate.index
    validate_scaled.columns = validate.columns

    # Test
    test_scaled.index = test.index
    test_scaled.columns = test.columns

    return train_scaled, validate_scaled, test_scaled


# In[156]:


get_ipython().system("git add 'prepare.ipynb'")


# In[157]:


get_ipython().system("git commit -m 'updates'")


# In[158]:


get_ipython().system('git push')


# In[ ]:




