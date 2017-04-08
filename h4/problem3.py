# CS 181, Spring 2017
# Homework 4: Clustering
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

class KMeans(object):
	# K is the K in KMeans
        def __init__(self, K):
		self.K = K

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	# Create Clusters Using Lloyd's Algorithm. 

	# Part 1: Initialize the Centers: 
	def __initializeCenterClusters(self):
		#size = self.X.shape[1]
		# Choose an initial mean randomly. Note that this is a vector.
		#initialmean = np.random.randn(self.K, size)

		self.means = np.random.randn(self.K, self.X.shape[1])
	# Helper Function for Update Mean.
	def __update(self):
		size = self.X.shape[0]
		distances = np.zeros((size,self.K))
		for k in range(self.K):
			distances[:,k] = np.sum(np.power(self.X - self.means[k],2))
		#Find the minimum distance, update based on that.
		#self.updates = np.zeros(self.K,size)
		# Update it based on the distances. The axis = 1 feature lets you do put it into the entire array at once.
		self.updates = np.argmin(distances,axis = 1)
	# Updates the mean
	def __updateMean(self):
		for k in range(self.K):
			self.means[k] = np.mean(self.X[self.updates == k])
	
	# Uses the elements to determine the objective function
	def __elements(self):
		size = self.X.shape[0]
		distances = np.zeros((size,self.K))
		for k in range(self.K):
			distances[:,k] = np.sum(np.power(self.X - self.means[k],2))
		elements = distances[np.arange(self.X.shape[0]), self.updates]
		return np.sum(elements)
	

	def fit(self, X):
		self.objective = []
		self.X = X
		self.__initializeCenterClusters()
		for k in range(10):
			self.__update()
			self.objective.append(self.__elements())
			self.__updateMean()
			self.objective.append(self.__elements())
		# This should be the value of the objective function.
		print self.__elements()


	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.means

	# This should return the arrays for D images from each cluster that are representative of the clusters.
	def get_representative_images(self, D):
		self.representative_images = np.zeros(self.K,D)
		distances = np.zeros((size,self.K))
		for k in range(self.K):
			distances[:,k] = np.power(self.X - self.means[k],2)
		for k in range(self.K):
			self.representative_images[k] = distances[:,k].argsort()[:D]
		return self.represenative_images


	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self):
		plt.figure()
		plt.plot(np.arange(len(self.objective)),  self.objective)
		plt.show()
		return
# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 1
KMeansClassifier = KMeans(K=1)
KMeansClassifier.fit(pics)
KMeansClassifier.create_image_from_array()



