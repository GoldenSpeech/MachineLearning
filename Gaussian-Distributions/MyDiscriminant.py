import numpy as np

class GaussianDiscriminant:
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance (S1=S2)
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance (S1!=S2)
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        class1 = []
        class2 = []
        rows = len(Xtrain)
        for j in range(self.d):
            for i in range(rows):
                if ytrain[i] == 1:
                   class1=Xtrain[ytrain == 1]
                else:
                    class2= Xtrain[ytrain == 2]
            self.mean[1][j] = np.mean(class2[:, j])##check here
            self.mean[0][j] = np.mean(class1[:, j])
        if self.shared_cov:
            # compute the class-independent covariance
            self.S =np.cov(np.transpose(Xtrain),ddof = 0)
        else:
            # compute the class-dependent covariance        
            self.S[0] = np.transpose(np.cov(np.transpose(class1),ddof = 0))
            self.S[1] =np.cov(np.transpose(class2),ddof = 0)

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            discriminantFunc1 = 0
            discriminantFunc2 = 0

            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    
                    if c == 0:
                        discriminantFunc1 = -0.5 * np.log(np.linalg.det(self.S)) - 0.5 * np.dot(np.dot(np.transpose(np.subtract(Xtest[i], self.mean[c])), np.linalg.inv(self.S)), np.subtract(Xtest[i], self.mean[c])) + np.log(self.p[c])
                    else:
                        discriminantFunc2 = -0.5 * np.log(np.linalg.det(self.S)) - 0.5 * np.dot(np.dot(np.transpose(np.subtract(Xtest[i], self.mean[c])), np.linalg.inv(self.S)), np.subtract(Xtest[i], self.mean[c])) + np.log(self.p[c]) 
                else:
                    if c == 0:
                        discriminantFunc1 = -0.5 * np.log(np.linalg.det(self.S[c])) - 0.5 * np.dot(np.dot(np.transpose(np.subtract(Xtest[i], self.mean[c])), np.linalg.inv(self.S[c])), np.subtract(Xtest[i], self.mean[c])) + np.log(self.p[c])
                    else:
                        discriminantFunc2 = -0.5 * np.log(np.linalg.det(self.S[c])) - 0.5 * np.dot(np.dot(np.transpose(np.subtract(Xtest[i], self.mean[c])), np.linalg.inv(self.S[c])), np.subtract(Xtest[i], self.mean[c])) + np.log(self.p[c]) 

            # determine the predicted class based on the values of discriminant function
            if discriminantFunc1 > discriminantFunc2:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d
    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        class1 = []
        class2 = []
        for j in range(self.d):
            for i in range(len(Xtrain)):
                if ytrain[i] == 1:
                   class1=Xtrain[ytrain == 1]
                else:
                    class2= Xtrain[ytrain == 2]
            self.mean[1][j] = np.mean(class2[:, j])
            self.mean[0][j] = np.mean(class1[:, j])

        # compute the variance of different features
        for v in range(self.d):
            self.S[v] = np.var(np.transpose(Xtrain)[v])

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            discriminantFunc1 = 0
            discriminantFunc2 = 0
            totalSum= 0
            for c in np.arange(self.k):
                totalSum = 0
                for k in range(self.d):
                    someX = Xtest[i][k] - self.mean[c][k]
                    totalSum = totalSum + (np.power(someX,2))/self.S[k]
                if c == 0:
                    discriminantFunc1 = -0.5 * totalSum + np.log(self.p[c])
                else:
                    discriminantFunc2 = -0.5 * totalSum + np.log(self.p[c])

            # determine the predicted class based on the values of discriminant function
            if discriminantFunc2 <discriminantFunc1:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
