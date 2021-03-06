#!/usr/bin/env python
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

tf.logging.set_verbosity(tf.logging.INFO)

print(tf.__version__)


# ## Read exploratory dataset into pandas dataframe

# In[ ]:

BASE_PATH = r'C:\Users\mistr\source\repos\rrmistry\kaggle\NY_Taxi_Cab' # os.path.dirname("__file__")
BASE_DATA_PATH = os.path.join(BASE_PATH, 'input\\train_split\\')

BATCH_SIZE = 4096
LEARNING_RATE = 0.00005
L1_REGULARIZATION_RATE = 0.5
L2_REGULARIZATION_RATE = 0.5
NUM_BUCKETS = 30
EVAL_START_DELAY = 30
EVAL_THROTTLE_DELAY = 120

print('Started reading dataset ------------- ', datetime.datetime.now())

# Try to load the data. This may be an intensive process
df_sample = pd.read_csv(os.path.join(BASE_DATA_PATH, 'train-000000000003.csv'), nrows=BATCH_SIZE * 4, parse_dates=["pickup_datetime"])

print('Finished reading dataset ------------- ', datetime.datetime.now())


# ## Describe some dataset statistics

# In[ ]:


df_sample.head(n=10)


# In[ ]:


df_sample.describe()


# ## Get stastics from training dataset


# In[ ]:


MEAN_PICKUP_LON = df_sample['pickuplon'].mean()
MEAN_PICKUP_LAT = df_sample['pickuplat'].mean()
MEAN_DROPOFF_LON = df_sample['dropofflon'].mean()
MEAN_DROPOFF_LAT = df_sample['dropofflat'].mean()

MAX_PICKUP_LON = df_sample['pickuplon'].max()
MAX_PICKUP_LAT = df_sample['pickuplat'].max()
MAX_DROPOFF_LON = df_sample['dropofflon'].max()
MAX_DROPOFF_LAT = df_sample['dropofflat'].max()

MIN_PICKUP_LON = df_sample['pickuplon'].min()
MIN_PICKUP_LAT = df_sample['pickuplat'].min()
MIN_DROPOFF_LON = df_sample['dropofflon'].min()
MIN_DROPOFF_LAT = df_sample['dropofflat'].min()


# ## Define training dataset properties

# In[ ]:


CSV_COLUMNS = 'key,key_original,fare_amount,pickup_datetime,dayofweek,dayofweek_numeric,hourofday,pickuplon,pickuplat,dropofflon,dropofflat,passengers'.split(',')
LABEL_COLUMN = 'fare_amount'
KEY_FEATURE_COLUMN = 'key'
DEFAULTS = [['nokey'], ['nokey'], [0.0], ['badDate'], ['Sun'], [0], [0], [-74.0], [40.0], [-74.0], [40.7], [0]]


# ## These are the raw input columns, and will be provided for prediction also

# In[ ]:


INPUT_COLUMNS = [
    # Define features
    tf.feature_column.categorical_column_with_vocabulary_list('dayofweek', vocabulary_list = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
    tf.feature_column.categorical_column_with_identity('hourofday', num_buckets = 24),

    # Numeric columns
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
    
    # Engineered features that are created in the input_fn
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean')
]


# ## Create feature engineering function that will be used in the input and serving input functions

# In[ ]:


def add_engineered(features):
    # this is how you can do feature engineering in TensorFlow
    lat1 = features['pickuplat']
    lat2 = features['dropofflat']
    lon1 = features['pickuplon']
    lon2 = features['dropofflon']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)

    # set features for distance with sign that indicates direction
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    dist = tf.sqrt(latdiff * latdiff + londiff * londiff)
    features['euclidean'] = dist
    return features


# ## Create serving input function to be able to serve predictions

# In[ ]:


def serving_input_fn():
    feature_placeholders = {
        # All the real-valued columns
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS[2:7]
    }
    feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
    feature_placeholders['hourofday'] = tf.placeholder(tf.int32, [None])

    features = add_engineered(feature_placeholders.copy())
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


# ## Create input function to load data into datasets

# In[ ]:


def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        
        def parse_dataset(filename, header_lines = 1):
            return tf.data.TextLineDataset(filenames=filename).skip(header_lines)
        
        def parse_batch(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label

        # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
        filenames_dataset = tf.data.Dataset.list_files(filename)
        
        # Read lines from text files
        dataset = filenames_dataset.flat_map(parse_dataset)
        
        # Parse text lines as comma-separated values (CSV)
        dataset = dataset.map(parse_batch)
        
        # Note:
        # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename  -> text lines)
        # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 10 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        return batch_features, batch_labels
    return _input_fn


# ## Define evaluation metrics

# In[ ]:


def add_eval_metrics(labels, predictions):
    pred_values = predictions['predictions']
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)
    }


# ## Build the estimator

# In[ ]:


def build_estimator(model_dir, hidden_units):
    """
     Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model.
    """

    # Input columns
    (dayofweek, hourofday, plat, plon, dlat, dlon, pcount, latdiff, londiff, euclidean) = INPUT_COLUMNS

    nbuckets = NUM_BUCKETS

    # Bucketize passengers
    b_passengers = tf.feature_column.bucketized_column(pcount, np.linspace(1, 8, 8).tolist())

    # Bucketize the lats & lons
    latbuckets = np.linspace(MIN_PICKUP_LAT, MAX_PICKUP_LAT, nbuckets).tolist()
    lonbuckets = np.linspace(MIN_PICKUP_LON, MAX_PICKUP_LON, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4 )
    day_hr =  tf.feature_column.crossed_column([dayofweek, hourofday], 24 * 7)

    # Wide columns and deep columns.
    wide_columns = [
        # Feature crosses
        dloc, ploc, pd_pair,
        day_hr,

        # Sparse columns
        dayofweek, hourofday,

        # Anything with a linear relationship
        b_passengers
    ]

    deep_columns = [
        # Embedding_column to "group" together ...
        tf.feature_column.embedding_column(pd_pair, nbuckets * 2),
        tf.feature_column.embedding_column(day_hr, nbuckets * 2),

        # Numeric columns
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean
    ]

    optimizor = tf.train.ProximalGradientDescentOptimizer(learning_rate = LEARNING_RATE,
                                                          l1_regularization_strength=L1_REGULARIZATION_RATE,
                                                          l2_regularization_strength=L2_REGULARIZATION_RATE,
                                                          ) # note the learning rate

    ## setting the checkpoint interval to be much lower for this task
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = 30, 
                                        keep_checkpoint_max = 10
                                        )

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = model_dir,
        dnn_activation_fn=tf.nn.relu,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_optimizer=optimizor,
        dnn_hidden_units = hidden_units,
        config = run_config)

    # add extra evaluation metric for hyperparameter tuning
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)
    return estimator


# ## Create estimator train and evaluate function

# In[ ]:


def train_and_evaluate(args):
    
    estimator = build_estimator(args['output_dir'], args['hidden_units'].split(' '))
    
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(
            filename = args['train_data_paths'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(
            filename = args['eval_data_paths'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        exporters = exporter,
        steps=100,
        start_delay_secs = EVAL_START_DELAY,    # start evaluating after N seconds
        throttle_secs = EVAL_THROTTLE_DELAY,    # evaluate every N seconds
        )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# In[ ]:


OUTPUTDIR = os.path.join(BASE_PATH, 'ML_Model')

if os.path.exists(OUTPUTDIR):
    shutil.rmtree(OUTPUTDIR,ignore_errors=True)

with tf.Session() as sess:
    
    arguments = {
        "output_dir": OUTPUTDIR,
        "train_data_paths": os.path.join(BASE_DATA_PATH, '*.csv'),
        "eval_data_paths": os.path.join(BASE_DATA_PATH, '../test-*.csv'),
        "train_batch_size": BATCH_SIZE,
        "eval_batch_size": 512,
        "train_steps": 1000000,
        "eval_steps": 10,
        "hidden_units": "2048 1024 512 512 256 128 64 32 16 8 4",
        "eval_delay_secs": 10,
        "min_eval_frequency": 1,
        "format": "csv"
    }
    
    # Run the training job:
    try:
        sess.run(tf.global_variables_initializer())
        
        train_and_evaluate(arguments)
        # print_datasets(arguments, sess)
    except:
        traceback.print_exc()


# In[ ]:




