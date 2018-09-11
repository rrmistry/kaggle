
# coding: utf-8

# Rohit's First Kernal - NYC Taxi Fare Prediction
# ===========
# This is the first kernal for submission for Google Cloud Playground [New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)
# 
# Strategy
# --------------------
# 1. Filter out outliers
#     1. Remove data outside NYC
#     2. Remove data where fare is unresonable (too high / too low)
# 2. Use Linear Regression ML Model On Clean Data
# 3. Use Linear Fit On Unclean Data
# 
# Using NYC Open Data
# -------------------
# NYC Open Data is stored in Google Big Query open datasets. To access this data in your notebook, check out kernal [How to Query the NYC Open Data
# ](https://www.kaggle.com/paultimothymooney/how-to-query-the-nyc-open-data)
# 

# In[94]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# to plot 3d scatter plots
from mpl_toolkits.mplot3d import Axes3D

import math

# to print out current time
import datetime
import os

import tensorflow as tf
import shutil
print(tf.__version__)


# In[95]:


#disable GPU for now
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# In[96]:


BATCH_SIZE = 5

# Try to load the data. This may be an intensive process
df_train = pd.read_csv(r'M:\kaggle\NY Taxi Cab\input\train.csv', nrows = 100, parse_dates=["pickup_datetime"])


# In[97]:


df_train.head(n=100)


# In[98]:


df_train.describe()


# In[115]:


CSV_COLUMNS = ['key',
               'fare_amount',
               'pickup_datetime',
               'pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'passenger_count']

LABEL_COLUMN = 'fare_amount' # 'pickup_datetime' #

DEFAULTS = [['NoKey'],
            [0.0],
            ['BadDate'],
            [-74.0],
            [40.0],
            [-74.0],
            [40.7],
            [1.0]]

TRAIN_TEST_SPLIT_RATIO = 0.8

def read_dataset(filename, mode, header_lines = 1):

    dataset = tf.data.TextLineDataset(filenames=filename).skip(header_lines)

    def parse_batch(single_batch):
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            
            # Test dataset does not have Label Column
            columns = tf.decode_csv(single_batch, record_defaults = DEFAULTS[:1] + DEFAULTS[2:])
            features = dict(zip(CSV_COLUMNS[:1] + CSV_COLUMNS[2:], columns))
            label = None # features.pop(LABEL_COLUMN) #
            
        else:
            
            columns = tf.decode_csv(single_batch, record_defaults = DEFAULTS)
            
            split_location = int(BATCH_SIZE * TRAIN_TEST_SPLIT_RATIO)
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                columns = map(lambda c: tf.slice(c, 0, 2), columns)
                
            elif mode == tf.estimator.ModeKeys.EVAL:
                columns = map(lambda c: tf.slice(c, 0, 2), columns)
            
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            
        return features, label

    # Shard dataset into batches
    dataset = dataset.batch(BATCH_SIZE)
    
    # Parse batches individually
    dataset = dataset.map(parse_batch)
                                                                                  
    return dataset

def get_train():
  return read_dataset('../input/train.csv', mode = tf.estimator.ModeKeys.TRAIN)

def get_valid():
  return read_dataset('../input/train.csv', mode = tf.estimator.ModeKeys.EVAL)

def get_test():
  return read_dataset('../input/test.csv', mode = tf.estimator.ModeKeys.PREDICT)


# In[116]:


csv_filename_ = '../input/rohit.csv'
header_lines_ = 1
delim_ = ','

with tf.Session() as tld_sess:
    tld_sess.run(tf.global_variables_initializer())
    #try:
    print('--- TRAIN DATASET -------------------')

    tld = get_train()
    tld_next = tld.make_one_shot_iterator().get_next()
    for x in range(BATCH_SIZE-1):
        tld_features, tld_label = tld_sess.run(tld_next)
        #print('-----------------------')
        #print(tld_features)
        print('-----------------------')
        print(tld_label)
        print('-----------------------')

    print('--- TEST DATASET --------------------')
    tlt = get_valid()
    tlt_next = tlt.make_one_shot_iterator().get_next()
    for x in range(BATCH_SIZE-1):
        tld_features, tld_label = tld_sess.run(tlt_next)
        #print('-----------------------')
        #print(tld_features)
        print('-----------------------')
        print(tld_label)
        print('-----------------------')
    #except Exception as e:
    #    print('Failed: '+ str(e))

