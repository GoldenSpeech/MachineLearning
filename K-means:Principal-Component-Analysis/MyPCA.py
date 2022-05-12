import numpy as np

class PCA():
    def __init__(self,num_dim=None):
        self.num_dim = num_dim
        self.mean = np.zeros([1,784]) # means of training data
        self.W = None # projection matrix

    def fit(self,X):
        # normalize the data to make it centered at zero (also store the means as class attribute)
        self.mean = np.mean(X, axis = 0)
        centered = X - self.mean

        # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
        coveran = np.cov(np.transpose(centered))

        val, vec = np.linalg.eigh(coveran)

        sortIndex = np.argsort(val)[::-1]
        sortValues = val[sortIndex]##checkback
        vec = np.transpose(vec)
        sortVec = []

        for j in range(len(sortIndex)):
            sortVec.append(vec[sortIndex[j]])
        self.num_dim = 784 
        if self.num_dim > 1: # is none#########
            # select the reduced dimension that keep >90% of the variance
            self.num_dim = 1
            sum = np.sum(sortValues[:self.num_dim])

            # store the projected dimension
            projDim = np.sum(sortValues)
            while (sum/np.sum(sortValues)) < 0.90 and self.num_dim <= len(X[0]):
                sum = np.sum(sortValues[:self.num_dim])
                self.num_dim += 1
                
            self.num_dim -= 1
            #self.num_dim = 784 # placeholder

        # determine the projection matrix and store it as class attribute
        self.W = [] #####is none #### placeholder

        for k in range(self.num_dim):
            self.W.append(sortVec[k])

        # project the high-dimensional data to low-dimensional one
        X_pca = np.dot(centered, np.transpose(self.W))    # placeholder

        return X_pca, self.num_dim

    def predict(self,X):
        # normalize the test data based on training statistics
        centered = X - self.mean

        # project the test data
        X_pca = np.dot(centered, np.transpose(self.W)) # placeholder

        return X_pca

    def params(self):
        return self.W, self.mean, self.num_dim
