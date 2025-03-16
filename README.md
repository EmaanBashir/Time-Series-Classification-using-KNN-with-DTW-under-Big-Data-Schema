# Time Series Classification using KNN with DTW

## Overview
This project explores time series classification using the K-Nearest Neighbors (KNN) algorithm with Dynamic Time Warping (DTW) as a distance measure. The study focuses on leveraging big data methodologies, specifically Apache Spark, to optimize KNN's performance and handle large-scale data efficiently. The dataset used is the UCI Human Activity Recognition (HAR) dataset. A shared Databricks cluster was used for this project to handle distributed computations efficiently.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Methodology](#methodology) 
- [Challenges Faced](#challenges-faced) 
- [Results](#results) 

- [Results](#results)
- [References](#references)

## Technologies Used
- Python
- PySpark
- Apache Spark (RDDs, DataFrames, Broadcast Variables)
- FastDTW (Dynamic Time Warping)
- NumPy
- Pandas
- Hive Metastore
- Databricks (Shared Cluster)

## Dataset
The dataset consists of sensor data collected from 30 participants performing six different activities while wearing a smartphone. Each sample contains 561 features representing accelerometer and gyroscope readings.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.8+
- Apache Spark 3.0+
- PySpark
- NumPy
- FastDTW
- Pandas

```sh
pip install pyspark fastdtw numpy pandas
```

## Methodology 
The project implements KNN classification using two distance measures: Euclidean distance and Dynamic Time Warping (DTW). Since KNN is a lazy learner that computes distances on demand, we focused on optimizing the computations using PySpark's distributed framework. 
### Model 1: Divide & Conquer (RDDs and FastDTW) 
- Uses Spark RDDs to partition data and apply KNN in parallel. 
- Computes Euclidean and FastDTW distances in a distributed fashion. 
- Offers scalability for large datasets by reducing redundant computations.

![image](https://github.com/user-attachments/assets/e84ac9a2-046d-482e-aa59-d1f341be7fe3)
  
### Model 2: Broadcasting (MapReduce & FastDTW) 
- Uses Spark’s broadcast variables to store training data on each worker node. 
- Reduces network overhead by minimizing data transfers. 
- Implements a MapReduce approach for efficient distance calculations.

![image](https://github.com/user-attachments/assets/39998828-528a-4a28-8456-680976e49695)
  
## Challenges Faced 
1. **Computational Cost**: Traditional DTW calculations required over a day for large datasets. Optimized FastDTW helped mitigate this. 
2. **Shared Cluster Limitations**: Running experiments on a shared Databricks cluster introduced execution time fluctuations. 
3. **Misclassification Issues**: Similar activities (e.g., walking upstairs vs. walking downstairs) posed classification challenges. 
4. **High Dimensionality**: The dataset's 561 features increased memory usage, requiring careful optimization. 
## Results 
The results demonstrate that using FastDTW improves classification accuracy and reduces computational overhead compared to traditional Euclidean distance measures. 
- **Computation Time**: Traditional DTW-based KNN took over 1.5 days, while optimized FastDTW KNN reduced this to ~16 minutes. 
- **Accuracy**: DTW-based KNN outperformed Euclidean KNN in classification accuracy for varying k values. 
- **Efficiency Gains**: Using Spark’s distributed processing reduced computation time by nearly 40% compared to non-PySpark implementations. 

## References
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- Salvador, S., & Chan, P. (2007). "Toward accurate dynamic time warping in linear time and space."
- Mahato, V., O'Reilly, M., & Cunningham, P. (2018). "A Comparison of k-NN Methods for Time Series Classification and Regression."

---

For further details, refer to the `Report.pdf` and `Presentation.pptx` files in the repository. 

