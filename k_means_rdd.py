#!/usr/bin/env python
# coding: utf-8

# In[38]:


from pyspark.sql import SparkSession
from pyspark import SparkContext
import numpy as np
from scipy.spatial.distance import euclidean


# In[39]:


spark = SparkSession.builder.getOrCreate()


# In[102]:


class Kmeans():
    '''
    RDD based spark estimator for K-Means clustering
    
    Attributes:
    k: Number of clusters
    max_iter: Total number of iteration to be performed
    rdd: Dataset with features in the form of RDD
    
    '''
    
    def __init__(self,k,max_iter):
        self.k = k
        self.max_iter = max_iter
        
    def initialize_cluster_centroids(self,rdd):
        
        cluster_centre = spark.sparkContext.parallelize(rdd.takeSample(False,self.k))
        cluster_number_rdd = spark.sparkContext.parallelize(np.arange(self.k))
        centroids = cluster_number_rdd.zip(cluster_centre)
        return centroids
        
    def cluster_assignment_step(self,rdd,initial_centroids):
        centroid_list = initial_centroids.collect()

        
        def nearest_cluster(x):
            lower_bound = np.inf
            nearest_cluster = 0
            for cluster,cluster_centre in x[1:]:
                distance = euclidean(cluster_centre,x[0])
                if distance < lower_bound:
                    nearest_cluster = cluster
                    lower_bound = distance
            return [nearest_cluster,x[0]]
        
        
        point_clusters = rdd.map(lambda x: list([np.array(x)])+(centroid_list))
        assignment_cluster = point_clusters.map(nearest_cluster)
        return assignment_cluster
        
        
    def fit(self,rdd):
        
        initial_centroids = self.initialize_cluster_centroids(rdd)
        centroids = initial_centroids
        self.converged_ = 'False'
        self.n_iter = 0
        
        for n_iter in range(self.max_iter):
            self.n_iter +=1
            
            # Cluster assignment_step
            cluster_assignment = self.cluster_assignment_step(rdd,centroids)
            
            # Cluster_centre update step
            centroids = cluster_assignment.reduceByKey(lambda a,b: np.mean(np.array([a,b]),axis = 0))
            
            new_cluster_assignment = self.cluster_assignment_step(rdd,centroids)
            if new_cluster_assignment.countByKey()==cluster_assignment.countByKey():
                self.converged_ = 'True'
                print('Converged')
                break
        
        if not self.converged_:
            print('Change the initialisation parameters or increase the max_iter')
        
        return new_cluster_assignment
                
            
            
            
            
        
        
        
        
        
    

