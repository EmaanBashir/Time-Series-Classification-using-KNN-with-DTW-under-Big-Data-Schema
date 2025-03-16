# Databricks notebook source
# MAGIC %md
# MAGIC # Data Pre-Processing
# MAGIC
# MAGIC
# MAGIC The data is in a test file with each row seperated by comma as string variable type.In each row, the values are seperated by space.The values repersent the time series value at time step t, where t is the index starting from 1 to 561. Hence each time series (each row) has 561 points/features.
# MAGIC The data pre-processing goal is to:
# MAGIC - split the rows, extract the values and convert them to float.
# MAGIC - make each value has a feature column, (Example Row 1 is time series 1 with 561 features)
# MAGIC - Vectorize features using Vectorizer and assembler.
# MAGIC - Add class labels for respective series.
# MAGIC
# MAGIC

# COMMAND ----------

!pip install fastdtw

# COMMAND ----------

# Import Required Libraries 
import pyspark.sql.functions as SQL_F
import pyspark.sql.types as SQL_T
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import col, udf, monotonically_increasing_id, abs,row_number,lit,split
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructField, StructType
from fastdtw import fastdtw
import time


# COMMAND ----------


# Initialize Spark Session
spark = SparkSession.builder \
    .appName("UCI HAR KNN-DTW Classification - Data Preprocessing") \
    .getOrCreate()

# Load data into Spark DataFrames
df_train_features = spark.table("hive_metastore.default.uci_har_x_train").withColumnRenamed("value", "train_features")
df_train_labels = spark.table("hive_metastore.default.uci_har_y_train").withColumnRenamed("value", "train_labels")
df_test_features =  spark.table("hive_metastore.default.uci_har_x_test").withColumnRenamed("value", "test_features")
df_test_labels =  spark.table("hive_metastore.default.uci_har_y_test").withColumnRenamed("value", "test_labels")


# udf to clean array of null values after split 
def clean_array(arr):
    return [x for x in arr if x.strip() != '']
  
# Register the UDF with Spark
clean_array_udf = SQL_F.udf(clean_array, SQL_T.ArrayType(SQL_T.StringType()))

# converting a single string of space seperated numbers into array of numbers in string format. 
df_train_features = df_train_features.select(SQL_F.split(df_train_features['train_features'], ' ').alias('split_array'))
df_test_features = df_test_features.select(SQL_F.split(df_test_features['test_features'], ' ').alias('split_array'))

# Apply the clean_array UDF to the 'split_array' column
df_train_features = df_train_features.withColumn('split_array', clean_array_udf(SQL_F.col('split_array')))
df_test_features = df_test_features.withColumn('split_array', clean_array_udf(SQL_F.col('split_array')))

# Calculate the number of columns to create.
max_cols = df_train_features.select('split_array').rdd.map(lambda x: len(x[0])).max()

# create columns for each feature point.
for i in range(max_cols):
    df_train_features = df_train_features.withColumn(f'feature_{i+1}',
                       SQL_F.when(SQL_F.col('split_array').getItem(i) == ' ', None)  # Replacing empty space with None
                       .otherwise(SQL_F.col('split_array').getItem(i)))
    df_test_features = df_test_features.withColumn(f'feature_{i+1}',
                       SQL_F.when(SQL_F.col('split_array').getItem(i) == ' ', None)  # Replacing empty space with None
                       .otherwise(SQL_F.col('split_array').getItem(i)))
# drop temporary columns
df_train_features = df_train_features.drop('split_array')
df_test_features = df_test_features.drop('split_array')

#display 
df_train_features.display()
df_test_features.display()

# Data Type changed from string to float
df_train_features = df_train_features.select([SQL_F.col(c).cast("float").alias(c) for c in df_train_features.columns])
df_train_labels = df_train_labels.select([SQL_F.col(c).cast("float").alias(c) for c in df_train_labels.columns])
df_test_features = df_test_features.select([SQL_F.col(c).cast("float").alias(c) for c in df_test_features.columns])
df_test_labels = df_test_labels.select([SQL_F.col(c).cast("float").alias(c) for c in df_test_labels.columns])


# Adding row_id to uniquely identify rows
df_train_features = df_train_features.alias("features")

w = Window().orderBy(lit('A'))
df_train = df_train_features.withColumn("row_id", row_number().over(w)).alias("features")
df_label = df_train_labels.withColumn("row_id", row_number().over(w)).alias("label")
df_test = df_test_features.withColumn("row_id", row_number().over(w)).alias("features")
df_label_test = df_test_labels.withColumn("row_id", row_number().over(w)).alias("label")

# Join features and labels on the row ID
df_train_with_label = df_train.join(df_label, "row_id")
df_test_with_label = df_test.join(df_label_test, "row_id")

# Save as SQL for later access during model development
# df_train_with_label.write.format("parquet").saveAsTable("train_processed_HAR")
# df_test_with_label.write.format("parquet").saveAsTable("test_processed_HAR")

# Vector Assembler
assembler = VectorAssembler(inputCols=df_train_with_label.columns[1:-1], outputCol="features")
train_features = assembler.transform(df_train_with_label).select("features","train_labels","row_id")
test_features = assembler.transform(df_test_with_label).select("features","test_labels","row_id")

train_features.display()
test_features.display()
spark.stop()


# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC # Model Development
# MAGIC
# MAGIC The project aims to implemment KNN with DTW using big data methologies. There is no KNN model in pyspark MLLib since,KNN is hard to parallelize in Spark because KNN is a "lazy learner" and the model itself is the entire dataset. On that line, we wanted to explore if it could actually be parallaized using different big data methologies, to speed up prediction than traditional non-pyspark implemmentation.
# MAGIC In addition, a comparision between euclidean and DTW(Dynamic Time Wraping) distance measure with KNN.

# COMMAND ----------

# Short pre-processing

# Load Processed data
dt_train = spark.table("hive_metastore.default.train_processed_HAR")
dt_test = spark.table("hive_metastore.default.test_processed_HAR")

# Vector Assembler
assembler = VectorAssembler(inputCols=dt_train.columns[1:-1], outputCol="features")
train_features = assembler.transform(dt_train).select("features","train_labels","row_id").cache()
test_features = assembler.transform(dt_test).select("features","test_labels","row_id").cache()

train_features.display()
test_features.display()


# COMMAND ----------

print(train_features.count())
print(test_features.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## MODEL IMPLEMMENTATION OK KNN with EUCLIDEAN

# COMMAND ----------

# KNN with Euclidean on test

import numpy as np
from collections import Counter
import heapq
      
# find most common class in k closest points
def most_common(labels):  
    return Counter(labels).most_common(1)[0][0]

# Calculate Euclidean distance between points 
def knnEuclidean(partition, k):
  partition_list = list(partition)
  train_points = []
  test_points = []

  distances = {}
  for train_point, test_point in partition_list:
    dist = (float(np.linalg.norm(np.array(test_point[0]) - np.array(train_point[0]))), train_point[1])
    if test_point not in distances:
      distances[test_point] = []
    if dist[0] != 0:
      distances[test_point].append(dist)
  
  for test_point, dists in distances.items():
    distances[test_point] = heapq.nsmallest(k, dists)

  results = [(test_point[0], dists) for test_point, dists in distances.items()]  
  return results

# KNN classifier to with Euclidean distance implemmentation, calculate distance between each point in test data and all other points in training data using map and mapPartitions on rdd with repartition
def calculateKnnEuclidean(k, tableName, trainPartitions = 100, testPartitions = 50):
  train_rdd = train_features.rdd.map(lambda row: (row['features'], row['train_labels'])).repartition(trainPartitions).cache()
  test_rdd = test_features.rdd.map(lambda row: (row['features'], row['test_labels'])).repartition(testPartitions).cache()

  res = train_rdd.cartesian(test_rdd).mapPartitions(lambda partition: knnEuclidean(partition, k))

  grouped_rdd = res.groupByKey()
  grouped_rdd = grouped_rdd.mapValues(lambda values: [item for sublist in values for item in sublist])

  min_values_rdd = grouped_rdd.mapValues(lambda values: heapq.nsmallest(k, values))
  predicted_labels = min_values_rdd.mapValues(lambda values: most_common([label for _, label in values])).cache()

  predicted_labels = predicted_labels.toDF(['Test_point', 'Predicted_label']).cache()
  predicted_labels.write.format("parquet").saveAsTable(tableName)


# COMMAND ----------

# Accuracy Calculation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import abs

def getAccuracy(predictedTable):
  # retrieve results
  predictions = spark.table(predictedTable).cache()
  actual_predictions = spark.table("hive_metastore.default.test_processed_HAR").cache()

  # Add vector column to test data
  assembler = VectorAssembler(inputCols=actual_predictions.columns[1:-1], outputCol="features")
  test_features = assembler.transform(actual_predictions).select("features","test_labels","row_id")

  # Join test data and predictions
  joined = predictions.join(test_features, predictions["Test_point"] == test_features["features"], how="inner")

  # Accuracy
  joined = joined.withColumn("abs_diff", abs(joined["Predicted_label"] - joined["test_labels"]))
  
  correct_predictions = joined.filter(joined["abs_diff"] == 0).count()
  total_predictions = joined.count()
  accuracy = correct_predictions / total_predictions

  return accuracy

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameter tuning of k 
# MAGIC (taking into account - optimal value of k is square root of train_samples (sqrt(7352)=85))

# COMMAND ----------

# KNN Euclidean for k = 5
k = 5
calculateKnnEuclidean(k, "hive_metastore.default.euclidean_predictions_HAR_k5")

# COMMAND ----------

pred = spark.table("hive_metastore.default.euclidean_predictions_HAR_k5")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.euclidean_predictions_HAR_k5")
print('Euclidean Accuracy for k = 5:', accuracy)

# COMMAND ----------

# KNN Euclidean for k = 10
k = 10
calculateKnnEuclidean(k, "hive_metastore.default.euclidean_predictions_HAR_k10")

# COMMAND ----------

pred = spark.table("hive_metastore.default.euclidean_predictions_HAR_k10")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.euclidean_predictions_HAR_k10")
print('Euclidean Accuracy for k = 10:', accuracy)

# COMMAND ----------

# KNN Euclidean for k = 20
k = 20
calculateKnnEuclidean(k, "hive_metastore.default.euclidean_predictions_HAR_k20")

# COMMAND ----------

pred = spark.table("hive_metastore.default.euclidean_predictions_HAR_k20")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.euclidean_predictions_HAR_k20")
print('Euclidean Accuracy for k = 20:', accuracy)

# COMMAND ----------

# KNN Euclidean for k = 50
k = 50
trainPartitions = 50
testPartitions = 50
calculateKnnEuclidean(k, "hive_metastore.default.euclidean_predictions_HAR_k50", trainPartitions, testPartitions)

# COMMAND ----------

pred = spark.table("hive_metastore.default.euclidean_predictions_HAR_k50")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.euclidean_predictions_HAR_k50")
print('Euclidean Accuracy for k = 50:', accuracy)

# COMMAND ----------

# KNN Euclidean for k = 70
k = 70
trainPartitions = 50
testPartitions = 50
calculateKnnEuclidean(k, "hive_metastore.default.euclidean_predictions_HAR_k70", trainPartitions, testPartitions)

# COMMAND ----------

pred = spark.table("hive_metastore.default.euclidean_predictions_HAR_k70")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.euclidean_predictions_HAR_k70")
print('Euclidean Accuracy for k = 70:', accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MODEL IMPLEMMENTATION WITH DTW

# COMMAND ----------

# KNN with DTW on test

from fastdtw import fastdtw
import numpy as np
from collections import Counter
import heapq

# calculate distance using fastdtw 
def dtw_distance(series1, series2):
    distance, path = fastdtw(series1, series2)
    return distance
# find most common class in k closest points
def most_common(labels):
    return Counter(labels).most_common(1)[0][0]
    
# Applying dtw for each test point vs all train points
def knnDTW(partition, k):
  partition_list = list(partition)
  train_points = []
  test_points = []

  distances = {}
  for train_point, test_point in partition_list:
    dist = (dtw_distance(np.array(test_point[0]), np.array(train_point[0])), train_point[1])
    if test_point not in distances:
      distances[test_point] = []
    if dist[0] != 0:
      distances[test_point].append(dist)
  
  for test_point, dists in distances.items():
    distances[test_point] = heapq.nsmallest(k, dists)

  results = [(test_point[0], dists) for test_point, dists in distances.items()]    
  return results


# KNN classifier to with FastDtw distance implemmentation, calculate distance between each point in test data and all other points in training data using map and mapPartitions on rdd with repartition  
def calculateKnnDTW(k, tableName, trainPartitions = 100, testPartitions = 50):
  train_rdd = train_features.rdd.map(lambda row: (row['features'], row['train_labels'])).repartition(trainPartitions).cache()
  test_rdd = test_features.rdd.map(lambda row: (row['features'], row['test_labels'])).repartition(testPartitions).cache()

  res = train_rdd.cartesian(test_rdd).mapPartitions(lambda partition: knnDTW(partition, k)).cache()

  grouped_rdd = res.groupByKey().cache()
  grouped_rdd = grouped_rdd.mapValues(lambda values: [item for sublist in values for item in sublist]).cache()

  min_values_rdd = grouped_rdd.mapValues(lambda values: heapq.nsmallest(k, values)).cache()
  predicted_labels = min_values_rdd.mapValues(lambda values: most_common([label for _, label in values])).cache()

  predicted_labels = predicted_labels.toDF(['Test_point', 'Predicted_label']).cache()
  predicted_labels.write.format("parquet").saveAsTable(tableName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameter tuning of k 
# MAGIC (taking into account - optimal value of k is square root of train_samples (sqrt(7352)=85))

# COMMAND ----------

# KNN DTW for k = 5
k = 5
calculateKnnDTW(k, "hive_metastore.default.fastdtw_test_predictions_HAR_k5")

# COMMAND ----------

pred = spark.table("hive_metastore.default.fastdtw_test_predictions_HAR_k5")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.fastdtw_test_predictions_HAR_k5")
print('DTW Accuracy for k = 5:', accuracy)

# COMMAND ----------

# KNN DTW for k = 10
k = 10
calculateKnnDTW(k, "hive_metastore.default.fastdtw_test_predictions_HAR_k10")

# COMMAND ----------

pred = spark.table("hive_metastore.default.fastdtw_test_predictions_HAR_k10")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.fastdtw_test_predictions_HAR_k10")
print('DTW Accuracy for k = 10:', accuracy)

# COMMAND ----------

# KNN DTW for k = 20
k = 20
calculateKnnDTW(k, "hive_metastore.default.fastdtw_test_predictions_HAR_k20")

# COMMAND ----------

pred = spark.table("hive_metastore.default.fastdtw_test_predictions_HAR_k20")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.fastdtw_test_predictions_HAR_k20")
print('DTW Accuracy for k = 20:', accuracy)

# COMMAND ----------

# KNN DTW for k = 50
k = 50
trainPartitions = 50
testPartitions = 50
calculateKnnDTW(k, "hive_metastore.default.fastdtw_test_predictions_HAR_k50", trainPartitions, testPartitions)

# COMMAND ----------

pred = spark.table("hive_metastore.default.fastdtw_test_predictions_HAR_k50")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.fastdtw_test_predictions_HAR_k50")
print('DTW Accuracy for k = 50:', accuracy)

# COMMAND ----------

# KNN DTW for k = 70
k = 70
trainPartitions = 50
testPartitions = 50
calculateKnnDTW(k, "hive_metastore.default.fastdtw_test_predictions_HAR_k70", trainPartitions, testPartitions)

# COMMAND ----------

pred = spark.table("hive_metastore.default.fastdtw_test_predictions_HAR_k70")
print(pred.count())
pred.display()

# COMMAND ----------

accuracy = getAccuracy("hive_metastore.default.fastdtw_test_predictions_HAR_k70")
print('DTW Accuracy for k = 70:', accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MODEL IMPLEMENTATION Using broadcast with dtw
# MAGIC and rdd, map and reduce to acheive parallization and optimization. 

# COMMAND ----------

# Import Required Libraries 
import pyspark.sql.functions as SQL_F
import pyspark.sql.types as SQL_T
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import col, udf, monotonically_increasing_id, abs,row_number,lit,split
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructField, StructType
from fastdtw import fastdtw
import time


# COMMAND ----------


start_time = time.time()

# Initialize Spark session
spark = SparkSession.builder.appName("KNN with DTW - Broadcast").getOrCreate()

# Define the schema for the data
schema = StructType([
    StructField("features", ArrayType(DoubleType()), True),
    StructField("label", IntegerType(), True)
])
# Load Preprocessed data 
dt_train = spark.table("hive_metastore.default.train_processed_HAR")
dt_test = spark.table("hive_metastore.default.test_processed_HAR")


# Vector Assembler
assembler = VectorAssembler(inputCols=dt_train.columns[1:-1], outputCol="features")
train_features = assembler.transform(dt_train).select("features","train_labels","row_id")
test_features = assembler.transform(dt_test).select("features","test_labels","row_id")


train_features_sub = train_features
test_features_sub = test_features


# Broadcast the train data
broadcast_train = spark.sparkContext.broadcast(train_features_sub.collect())

# Function to calculate DTW distance
def dtw_distance(series1, series2):
    distance, path = fastdtw(series1, series2)
    return distance

# function for KNN to calculate dtw distance between test series and all other series in broadcasted train dataset.
def knn(index_feature):
    index, feature = index_feature
    return [
        (index, (dtw_distance(np.array(feature), np.array(train_row['features'])), train_row['train_labels']))
        for train_row in broadcast_train.value
    ]
# RDD calculation using flatMap and reduceByKey
# Map to emit distances - using map to paralleize and reparitition of data to improve parallelisation.
test_distances_rdd = test_features_sub.rdd.map(lambda row: (row['row_id'], row['features'])).flatMap(knn).repartition(200)
# Reduce to find the nearest feature by distance (k=1)
min_distances_rdd = test_distances_rdd.reduceByKey(lambda a, b: a if a[0] < b[0] else b)

end_time = time.time()
print("Total Time:",end_time - start_time)


# COMMAND ----------

# create dataframe to store Predicted Labels and Test Labels to find accuracy
schema = SQL_T.StructType([
    SQL_T.StructField("row_id", SQL_T.IntegerType(), True),
    SQL_T.StructField("predicted_label", SQL_T.FloatType(), True)
])

df=min_distances_rdd.toDF()
df = df.withColumn("Min_distance", col("_2._1")) \
    .withColumn("Predicted_label", col("_2._2")) \
    .drop("_2")

joined_df = test_features_sub.join(df, test_features_sub.row_id == df._1, "inner").select(col('row_id'),col('test_labels'),col('Predicted_label'))

# joined_df.write.format("parquet").saveAsTable("test_predictions_HAR_dtw_broadcast_all")

# COMMAND ----------


# Calculate Accuracy
test_predictions = spark.table("hive_metastore.default.test_predictions_HAR_dtw_broadcast_all")
test_predictions = test_predictions.withColumn("abs_diff", abs(test_predictions["Predicted_label"] - test_predictions["test_labels"]))
correct_predictions = test_predictions.filter(test_predictions["abs_diff"] == 0).count()
total_predictions = test_predictions.count()
accuracy = correct_predictions / total_predictions

print("Accuracy - DTW with Broadcast k=1:", accuracy)
# spark.stop()


# COMMAND ----------

# MAGIC %md
# MAGIC
