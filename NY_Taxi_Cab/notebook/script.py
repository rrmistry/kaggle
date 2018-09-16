
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

# ## Setup Import Libraries

# In[ ]:


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


# ## Read exploratory dataset into pandas dataframe

# In[ ]:

BASE_PATH = os.path.dirname(__file__)

BATCH_SIZE = 512

print('Started reading dataset ------------- ', datetime.datetime.now())

# Try to load the data. This may be an intensive process
df_train = pd.read_csv(os.path.join(BASE_PATH, (r'..\input\train.csv')), nrows=BATCH_SIZE*2, parse_dates=["pickup_datetime"])

print('Finished reading dataset ------------- ', datetime.datetime.now())


# ## Describe some dataset statistics

# In[ ]:


df_train.head(n=10)


# In[ ]:


df_train.describe()


# ## Define training dataset properties

# In[ ]:


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


# ## Create feature engineering function that will be used in the input and serving input functions

# In[ ]:


def add_engineered(features):
    # this is how you can do feature engineering in TensorFlow
    lat1 = features['pickup_latitude']
    lat2 = features['dropoff_latitude']
    lon1 = features['pickup_longitude']
    lon2 = features['dropoff_longitude']
    lat_diff = (lat1 - lat2)
    lon_diff = (lon1 - lon2)
    
    # set features for distance with sign that indicates direction
    features['lat_diff'] = lat_diff
    features['lon_diff'] = lon_diff
    dist = tf.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
    features['euclidean'] = dist
    
    return features


# In[ ]:


def read_dataset(filenames, mode, batch_size = BATCH_SIZE):
        
    def _input_fn():
        
        def parse_dataset(filename, header_lines = 1):
            return tf.data.TextLineDataset(filenames=filename).skip(header_lines) 
        
        def parse_batch(value_column):
            if mode == tf.estimator.ModeKeys.PREDICT:
                columns = tf.decode_csv(value_column, record_defaults = DEFAULTS[:1] + DEFAULTS[1:])
                features = dict(zip(CSV_COLUMNS[:1] + CSV_COLUMNS[1:], columns))
                label = DEFAULTS[1]
            else:
                columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
                features = dict(zip(CSV_COLUMNS, columns))
                label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label

        # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
        filenames_dataset = tf.data.Dataset.list_files(filenames)
        
        # Read lines from text files
        dataset = filenames_dataset.flat_map(parse_dataset)
        
        # Parse text lines as comma-separated values (CSV)
        dataset = dataset.map(parse_batch)
        
        # Note:
        # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
        # use tf.data.Dataset.map            to apply one to one    transformations (here: text line -> feature list)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
                num_epochs = None # loop indefinitely
                dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
                num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        return dataset

    return _input_fn


# In[ ]:


def get_train():
    return read_dataset(os.path.join(BASE_PATH, r'../input/train/train-*.csv'), mode = tf.estimator.ModeKeys.TRAIN)

def get_valid():
    return read_dataset(os.path.join(BASE_PATH, r'../input/train/test-*.csv'), mode = tf.estimator.ModeKeys.EVAL)

def get_test():
    return read_dataset(os.path.join(BASE_PATH, r'../input/test.csv'), mode = tf.estimator.ModeKeys.PREDICT)


# In[ ]:


INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),
]

def add_more_features(feats):
    # Nothing to add (yet!)
    return feats

feature_cols = add_more_features(INPUT_COLUMNS)


# In[ ]:


def print_rmse(model, name, input_fn):
    metrics = model.evaluate(input_fn = input_fn, steps = None)
    print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))


# In[ ]:


OUTDIR = '../taxi_trained'

tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as sess:
    try:
        sess.run(tf.global_variables_initializer())

        if False:
            shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
            
            model = tf.estimator.LinearRegressor(feature_columns = feature_cols, model_dir = OUTDIR)
            
            print('Begin Training ---------------- ', datetime.datetime.now())
            model.train(input_fn = get_train(), steps = 1000)
            
            print('Begin Testing ---------------- ', datetime.datetime.now())
            print_rmse(model, 'validation', get_valid())
            
            print('Finished Testing ---------------- ', datetime.datetime.now())
        else:
            #try:
            print('--- TRAIN DATASET -------------------')

            tld = get_train()
            tld_next = tld().make_one_shot_iterator().get_next()
            for x in range(BATCH_SIZE-1):
                tld_features, tld_label = sess.run(tld_next)
                #print('-----------------------')
                #print(tld_features)
                print('-----------------------')
                print(tld_label)
                print('-----------------------')
    except:
        traceback.print_exc()

