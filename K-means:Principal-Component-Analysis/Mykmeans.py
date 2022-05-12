#import libraries
import numpy as np

class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None # centers for different clusters
        self.cluster_label = np.zeros([k,]) # class labels for different clusters
        self.error_history = []

    def fit(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005]

        for i in range(len(init_idx)):
            if i == 0:
                self.center = X[init_idx[i],:]
            else:
                self.center = np.vstack([self.center,X[init_idx[i],:]])

        num_iter = 0 # number of iterations for convergence

        # initialize the cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False

        # iteratively update the centers of clusters till convergence
        while not is_converged:

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                distance = np.linalg.norm(X[i,:] - self.center, axis = 1)

                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                cluster_assignment[i] = np.argmin(distance)


            # update the centers based on cluster assignment (M step)
            for k in range(self.num_cluster):
                self.center[k] = np.mean(X[cluster_assignment == k], axis = 0 ) 


            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1


        # compute the class label of each cluster based on majority voting (remember to update the corresponding class attribute)
        for k in range(self.num_cluster):
            vote = np.zeros([self.num_cluster + 2,])
            for i in range(len(X)):
                if cluster_assignment[i] == k:
                    ##return here
                    vote[int(y[i])] += 1
            
            self.cluster_label[k] = np.argmax(vote)

        return num_iter, self.error_history

    def predict(self,X):
        # predicting the labels of test samples based on their clustering results
        prediction = np.ones([len(X),]) # placeholder

        # iterate through the test samples
        for i in range(len(X)):
            # find the cluster of each sample
            distance_pred = np.linalg.norm(X[i,:] - self.center, axis = 1)
            selected = np.argmin(distance_pred)
            # use the class label of the selected cluster as the predicted class
            prediction[i] = self.cluster_label[selected]

        return prediction

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for j in range(len(X)):
            error += np.square(np.linalg.norm(X[j,:] - self.center[cluster_assignment[j]]))

        return error

    def params(self):
        return self.center, self.cluster_label
