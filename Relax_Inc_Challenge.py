#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


#read the user engagement data
engagement = pd.read_csv('/Users/kachu/Desktop/DS/relax_challenge/takehome_user_engagement.csv')
engagement.head()
engagement.info()


# In[3]:


#check how many users and how many logins
print('There were {} users and {} total logins'.format(
    engagement['user_id'].nunique(), len(engagement)
))


# In[6]:


# convert time_stamp into datetime format
engagement['time_stamp'] = pd.to_datetime(
    engagement['time_stamp'], format='%Y-%m-%d %H:%M:%S'
)
engagement.head()


# In[11]:


# define a function to see if a user logged in on 3 seperate days in a 7-day period.
def logins_in_days(df, days=7, logins=3):
    from datetime import timedelta
    
    # first drop duplicate days and sort by day
    df['date'] = df['time_stamp'].dt.date
    df = df.drop_duplicates(subset='date').sort_values('date')
    
    # calculate how many days has passed for every 3 logins
    passed_days = df['date'].diff(periods=logins-1)
    
    # check if any passed time is less than 7 days
    return any(passed_days <= timedelta(days=days))

adopted = engagement.groupby('user_id').apply(logins_in_days)
adopted.name = 'adopted_user'

print('There were {} adopted users out of {} users'.format(
    sum(adopted), len(adopted)))


# In[19]:


cols = ['object_id', 'creation_source', 'creation_time', 
        'last_session_creation_time', 'opted_in_to_mailing_list',
        'enabled_for_marketing_drip', 'org_id', 'invited_by_user_id']
users = pd.read_csv('/Users/kachu/Desktop/DS/relax_challenge/takehome_users.csv', usecols=cols)
users.head()


# In[ ]:




