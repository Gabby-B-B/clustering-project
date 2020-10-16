#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

from env import host, user, password


# In[2]:


def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[5]:


sql_query = ''' SELECT * FROM properties_2017
LEFT JOIN airconditioningtype ON properties_2017.airconditioningtypeid= airconditioningtype.airconditioningtypeid
LEFT JOIN architecturalstyletype ON properties_2017.architecturalstyletypeid = architecturalstyletype.architecturalstyletypeid
LEFT JOIN heatingorsystemtype ON properties_2017.heatingorsystemtypeid = heatingorsystemtype.heatingorsystemtypeid
LEFT JOIN propertylandusetype ON properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid
LEFT JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
LEFT JOIN buildingclasstype ON properties_2017.buildingclasstypeid = buildingclasstype.buildingclasstypeid
LEFT JOIN storytype ON properties_2017.storytypeid = storytype.storytypeid
LEFT JOIN typeconstructiontype ON properties_2017.typeconstructiontypeid = typeconstructiontype.typeconstructiontypeid
LEFT JOIN unique_properties ON properties_2017.parcelid = unique_properties.parcelid
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
AND properties_2017.propertylandusetypeid IN ('260', '261', '262', '263', '264', '266', '268', '273', '274', '275', '276', '279')
AND transactiondate BETWEEN '2017-04-30' AND '2017-07-01'
            '''
df = pd.read_sql(sql_query, get_connection('zillow'))
df.to_csv('zillow_df.csv')
df.info()


# In[7]:


def get_zillow_data():
    filename = "zillow_df.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('''SELECT * FROM properties_2017
LEFT JOIN airconditioningtype ON properties_2017.airconditioningtypeid= airconditioningtype.airconditioningtypeid
LEFT JOIN architecturalstyletype ON properties_2017.architecturalstyletypeid = architecturalstyletype.architecturalstyletypeid
LEFT JOIN heatingorsystemtype ON properties_2017.heatingorsystemtypeid = heatingorsystemtype.heatingorsystemtypeid
LEFT JOIN propertylandusetype ON properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid
LEFT JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
LEFT JOIN buildingclasstype ON properties_2017.buildingclasstypeid = buildingclasstype.buildingclasstypeid
LEFT JOIN storytype ON properties_2017.storytypeid = storytype.storytypeid
LEFT JOIN typeconstructiontype ON properties_2017.typeconstructiontypeid = typeconstructiontype.typeconstructiontypeid
LEFT JOIN unique_properties ON properties_2017.parcelid = unique_properties.parcelid
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
AND properties_2017.propertylandusetypeid IN ('260', '261', '262', '263', '264', '266', '268', '273', '274', '275', '276', '279')
AND transactiondate BETWEEN '2017-04-30' AND '2017-07-01' ''', get_connection('zillow'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_file(filename)
        # Return the dataframe to the calling code
        return df


# In[ ]:




