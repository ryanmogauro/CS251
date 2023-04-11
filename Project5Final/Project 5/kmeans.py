'''kmeans.py
Performs K-Means clustering
Ryan Mogauro
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from palettable import cartocolors


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        data_copy = self.data.copy()
        return data_copy

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distance =  np.sqrt(np.sum((pt_2 - pt_1)**2))
        return distance

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distances = [self.dist_pt_to_pt(x, pt) for x in centroids]
        return np.array(distances)

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        self.k = k   
        rand_rows = np.random.choice(self.data.shape[0], size=k, replace=False)
        initial_centroids = self.data[rand_rows, :]
        self.centroids = initial_centroids
        return initial_centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        
        self.initialize(k)
        
        iterations = 0
        previous_centroid = self.centroids
        self.data_centroid_labels = self.update_labels(previous_centroid)

        while iterations < max_iter:
            previous_centroid = self.centroids
            self.centroids, centroid_diff = self.update_centroids(k, self.data_centroid_labels, previous_centroid)
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.inertia = self.compute_inertia()
            iterations += 1
            if -tol < np.mean(centroid_diff) < tol:
                break
        return self.inertia, iterations
        
    

    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        best = 9223372036854775807
        cent = self.centroids
        labels = self.data_centroid_labels
        inert = self.inertia
        for i in range(n_iter):
            inertia, _ = self.cluster(k = k)
            if inertia < best:
                cent = self.centroids
                labels = self.data_centroid_labels
                inert = self.inertia
        self.centroids = cent
        self.data_centroid_labels = labels
        self.inertia = inert

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        pass
        
        labels = []
        for i in range(self.data.shape[0]):     
            distances = self.dist_pt_to_centroids(self.data[i], centroids)
            labels.append(np.argmin(distances))
        
        self.data_centroid_labels = np.array(labels)
        return np.array(labels)
    
    
    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        new_centroids = []
        
        for i in range(0,k):
            #if no points assigned to k
            if(i not in data_centroid_labels):
                rand_pt_index = np.random.randint(self.data.shape[0])
                rand_pt = self.data[rand_pt_index]
                new_centroids.append(rand_pt)
            else:
            #if i is in list
                #get all data points of corresponding centroid
                counter = 0
                centroid_data = []
                for x in data_centroid_labels:
                    if x == i:
                        centroid_data.append(self.data[counter])
                    counter+=1
                new_centroid = np.mean(centroid_data, axis = 0)
                new_centroids.append(new_centroid)
        
        self.centroids = np.array(new_centroids)
        
        diff = self.centroids - prev_centroids
        return (self.centroids, diff)

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        
        total = 0
        counter = 0
        for i in self.data_centroid_labels:
            centroid = self.centroids[i]
            point = self.data[counter]
            distance = self.dist_pt_to_pt(point, centroid)
            total += (distance**2)
            counter+=1
            
        average = float(total / len(self.data_centroid_labels))
        return average

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.data_centroid_labels,
                    cmap=cartocolors.sequential.agSunset_7.mpl_colormap)
        plt.scatter(
            self.centroids[:, 0], self.centroids[:, 1], marker="x", s=50, c="Black")
        plt.title("Clusters")
        plt.xlabel("X")
        plt.ylabel("Y")

    def elbow_plot(self, max_k):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        pass
        inertia = []
        clusters = [x for x in range(1,max_k+1)]
        for i in range(1, max_k+1):
            self.cluster_batch(k=i, n_iter=1)
            inertia.append(self.inertia)
        plt.plot(clusters, inertia, markersize=15)
        plt.xlabel("K")
        plt.ylabel("Inertia")
        plt.title("Inertia vs. K")
        plt.xticks(clusters)
        
        

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        data = self.get_data()
        for i in range(self.data.shape[0]):
            data[i] = self.centroids[self.data_centroid_labels[i]]
        self.data = data
