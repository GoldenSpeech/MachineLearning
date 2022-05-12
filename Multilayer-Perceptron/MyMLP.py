import numpy as np

class Normalization:
    def __init__(self,):
        self.mean = np.zeros([1,64]) # means of training features
        self.std = np.zeros([1,64]) # standard deviation of training features

    def fit(self,x):
        # compute the statistics of training samples (i.e., means and std)
        self.mean = np.mean(x, axis = 0)
        self.std = np.std(x, axis = 0)

    def normalize(self,x):
        # normalize the given samples to have zero mean and unit variance (add 1e-15 to std to avoid numeric issue)
        x = (x - self.mean) / (self.std + 1e-15) 

        return x

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for j in range(len(label)):  
        one_hot[j][label[j]] = 1

    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    x = np.clip(x,a_min=-100,a_max=100) # for stablility, do not remove this line

    f_x = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) # placeholder #####TRy replacing this with 1

    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    x = np.array(x)

    f_x = np.zeros([len(x), len(x[0])]) # placeholder

    theSUM = np.zeros([1, len(x)])
    for i in range(len(x[0])):
        theSUM +=  np.exp(x[:, i])
    for j in range(len(x[0])):
        f_x[:, j] = np.exp(x[:, j]) / theSUM

    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)

            zh = tanh(np.dot(train_x, self.weight_1) + self.bias_1)
            yh = softmax(np.dot(zh, self.weight_2 + self.bias_2))


            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters

            delW = np.zeros(self.weight_1.shape)
            delV = np.zeros(self.weight_2.shape)

            for i in range(yh.shape[0]):
                dEY = - train_y[i, :] / yh[i, :]
                dBW = train_x[i, :]
                dAZ = self.weight_2

                dZB = 1 - (zh[i, :]**2)
                dAV = zh[i, :]
                dYA = np.tile(yh[i, :], (yh.shape[1], 1)) * (np.identity(yh.shape[1]) - np.tile(np.transpose([yh[i, :]]), (1, yh.shape[1])))

                delW = delW- lr * np.transpose(np.dot(np.transpose(np.dot([np.dot(dEY, dYA)], np.transpose(dAZ)) * dZB), [dBW]))
                delV = delV- lr * np.transpose(np.dot(np.transpose([np.dot(dEY, dYA)]), [dAV]))


            #update the parameters based on sum of gradients for all training samples
            self.weight_1 = self.weight_1 + delW
            self.weight_2 = self.weight_2 + delV

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        weighy =np.dot(x, self.weight_1)
        zh = tanh( weighy+ self.bias_1)
        probability2 = softmax(np.dot(zh, self.weight_2) + self.bias_2)


        # convert class probability to predicted labels

        y = np.zeros([len(x),]).astype('int') # placeholder

        for j in range(len(x)):
            try:
                anArr = np.hstack(np.where(probability2[j] == probability2[j].max()))
                y[j] = anArr[0]
            except:
                anArr = np.array([0])
                y[j] = anArr[0]

        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        z = tanh(np.dot(x, self.weight_1) + self.bias_1)# placeholder
        
        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2