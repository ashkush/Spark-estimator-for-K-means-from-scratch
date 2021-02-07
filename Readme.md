This repository contains Spark estimator for Kmeans algorithms used for unsupervised clustering. Written in map reduce form from scratch, this estimator utilises Spark
cluster parallel processing power by distruting the dataset in the form of RDD on worker nodes of a spark cluster and performing map and reduce operation on each row of 
the RDD for the kmeans clustering.


Initialisation of the K-means clusters arer done by randomly selecting points within the dataset and no further changes possible for cluster assignment
is utilised as a convergence criterion for stopping the fitting process of the estimator.
