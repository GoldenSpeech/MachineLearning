import numpy as np

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node), -1 means the node is not a leaf node
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample till reaching a leaf node
            thisNode = self.root##here
            while(thisNode.left_child is not None or thisNode.right_child is not None):
                if(test_x[i][thisNode.feature] != 1):
                    thisNode = thisNode.left_child
                else:
                    thisNode = thisNode.right_child
                prediction[i] = thisNode.label##toHere
        return prediction

    def generate_tree(self,data,label):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node based on minimum node entropy (if yes, find the corresponding class label with majority voting and exit the current recursion)
        if(node_entropy <= self.min_entropy):##here
            cur_node.label = np.argmax(np.bincount(label))
            return cur_node##toHere


        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion
        lft = data[data[:, selected_feature] == 0]##here
        rht = data[data[:, selected_feature] == 1]
        left_label = label[data[:, selected_feature] == 0]
        right_label = label[data[:, selected_feature] == 1]


        cur_node.left_child = self.generate_tree(lft, left_label)
        cur_node.right_child = self.generate_tree(rht, right_label)##toHere

        return cur_node

    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        theEntropy = {}##here

        for i in range(len(data[0])):
            sampleL  = []##here
            sampleR = []
            for j in range(data.shape[0]):
                if(data[j][i] == 0):
                    sampleL.append(label[j])
                else:
                    sampleR.append(label[j])
            theEntropy[i] = self.compute_split_entropy(sampleL, sampleR)
            
            # compute the entropy of splitting based on the selected features
            best_feat = min(theEntropy, key=theEntropy.get)##toHere

            # select the feature with minimum entropy

        return best_feat

    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split (with compute_node_entropy function), left_y and right_y are labels for the two branches
        split_entropy = -1 # placeholder
        #here
        lefty=len(left_y) 
        righty =len(right_y)
        totalVaule = righty+ lefty
        leftnodeEntry=  ((len(right_y)/totalVaule) * self.compute_node_entropy(right_y))
        rightnodeEnt= ((len(left_y)/totalVaule)*self.compute_node_entropy(left_y))
        split_entropy = rightnodeEnt + leftnodeEntry
        #here
        return split_entropy

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        node_entropy = -1 # placeholder

        prob = {}
        for k in range(len(label)):
            
            if label[k] in prob.keys():
                prob[label[k]] += 1
            else:
                prob[label[k]] = 1
        theEntrypyofTreeNode = []


        for p in prob.keys():
            prob[p] = prob[p]/len(label)


            theEntrypyofTreeNode.append(-prob[p] * np.log2(prob[p] + 1e-15))
        node_entropy = sum(theEntrypyofTreeNode) # placeholder


        return node_entropy
