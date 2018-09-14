
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

# In[14]:


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

import traceback

import tensorflow as tf
import shutil
print(tf.__version__)


# In[15]:


#disable GPU for now
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# In[16]:


BATCH_SIZE = 10000

# Try to load the data. This may be an intensive process
df_train = pd.read_csv(r'M:\kaggle\NY Taxi Cab\input\train.csv', nrows = BATCH_SIZE, parse_dates=["pickup_datetime"]);


# In[17]:


df_train.head(n=10)


# In[18]:


df_train.describe()


# In[19]:


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


# In[20]:


def read_dataset(filename, mode, batch_size = 512):
  def _input_fn():
    def decode_csv(value_column):
      columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
      features = dict(zip(CSV_COLUMNS, columns))
      label = features.pop(LABEL_COLUMN)
      return features, label

    # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
    filenames_dataset = tf.data.Dataset.list_files(filename)
    # Read lines from text files
    textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
    # Parse text lines as comma-separated values (CSV)
    dataset = textlines_dataset.map(decode_csv)
    
    # Note:
    # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
    # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None # loop indefinitely
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    else:
        num_epochs = 1 # end-of-input after this

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    
    return dataset.make_one_shot_iterator().get_next()
  return _input_fn


# In[21]:


def get_train():
  return read_dataset('../input/train/train-*.csv', mode = tf.estimator.ModeKeys.TRAIN)

def get_valid():
  return read_dataset('../input/train/test-*.csv', mode = tf.estimator.ModeKeys.EVAL)

def get_test():
  return read_dataset('../input/test.csv', mode = tf.estimator.ModeKeys.PREDICT)


# In[23]:


with tf.Session() as sess:
    try:
        pass
    except:
        traceback.print_exc()

