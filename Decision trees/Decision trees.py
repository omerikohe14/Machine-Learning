import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 1.0
    num_Of_Instances = len(data)
    class_counts = np.unique(data[:, -1], return_counts=True)[1] / num_Of_Instances
    gini -= np.sum(class_counts**2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    num_Of_Instances = len(data)
    class_counts = np.unique(data[:, -1], return_counts=True)[1]
    for class_count in class_counts:
        proportion = class_count / num_Of_Instances
        entropy -= (proportion*(np.log2(proportion)))
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0.0
    information_Gain = impurity_func(data)
    split_Information = 0.0
    unique_values = np.unique(data[:, feature])
    groups = {value: data[data[:, feature] == value] for value in unique_values} 
    num_Of_Instances = len(data)
    for data_subset in groups.values():
        proportion = len(data_subset) / num_Of_Instances
        current_Impurity = impurity_func(data_subset)
        information_Gain -= proportion * current_Impurity
        split_Information -= proportion * np.log2(proportion)
    if gain_ratio:
        # Avoid 0 division
        if split_Information == 0:
            goodness = 0
        else:
            goodness = information_Gain / split_Information
    else:
        goodness = information_Gain
    return goodness, groups


class DecisionNode:
    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = True # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        # return the most freauent label
        unique_values , counts = np.unique(self.data[:, -1] , return_counts=True)
        most_Frequent_Index = np.argmax(counts)
        pred = unique_values[most_Frequent_Index] 
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        if self.depth == self.max_depth:
            return
        features_Goodness = {feature: goodness_of_split(
            self.data, feature, impurity_func, gain_ratio=self.gain_ratio) for feature in range(len(self.data[0])-1)}
        best_Feature_To_Split = max(features_Goodness, key=lambda k: features_Goodness[k][0])
        # check if all features did'nt improv the prediction
        if features_Goodness[best_Feature_To_Split][0] == 0:
            # if so look for thr first one that actually split the node
            for feature in range(len(features_Goodness)):
                if len(features_Goodness[feature][1]) > 1:
                    best_Feature_To_Split = feature
                    break
        self.feature = best_Feature_To_Split
        # check for chi pruning
        if self.chi != 1:
            chi = self.calc_Chi()
            degree_Of_Freedom = (len(features_Goodness[best_Feature_To_Split][1]) - 1) * (len(np.unique(self.data[:,-1])) - 1)
            # chi pruning condition if true dont split and stays a leaf
            if chi < chi_table[degree_Of_Freedom][self.chi]:
                self.feature = -1
                return 
        for key in features_Goodness[best_Feature_To_Split][1].keys():
            self.add_child(DecisionNode(data=features_Goodness[best_Feature_To_Split][1][key] ,depth=self.depth+1 ,max_depth = self.max_depth,chi=self.chi , gain_ratio=self.gain_ratio) , key)
        # after adding children the node is no longer a leaf
        self.terminal = False
        
    def is_Pure(self):
        return len(np.unique(self.data[:,-1])) == 1
    
    # recursive help method for tree building
    def build_tree_helper(self , impurity):
        if self.is_Pure():
            self.terminal = True
            return
        self.split(impurity)
        for child in self.children:
            child.build_tree_helper(impurity)

    # chi calculation for the current node split
    def calc_Chi(self):
        data = self.data
        feature = self.feature
        num_Of_Instances = len(data)
        chi = 0
        unique_values = np.unique(data[:, feature])
        labels = np.unique(data[:, -1])
        groups = {value: data[data[:, feature] == value] for value in unique_values}
        for label in labels:
            num_Of_Label_Instances = np.sum(data[:,-1] == label)
            P_Label = num_Of_Label_Instances / num_Of_Instances
            for feature_Value in groups.keys():
                Df = len(groups[feature_Value])
                p_Feature_And_Label = np.sum(groups[feature_Value][:,-1] == label)
                E_Label = Df * P_Label
                chi += (np.square((p_Feature_And_Label - E_Label))/E_Label)
        return chi

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data=data, gain_ratio=gain_ratio , chi=chi , max_depth=max_depth)
    root.build_tree_helper(impurity)  
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    if root.terminal:
        return root.pred
    instance_Feature_Value = instance[root.feature]
    # check if the instance attribute exists if not return the current node prediction
    try:
        instance_Feature_Value_Index = root.children_values.index(instance_Feature_Value)
    except:
        return root.pred
    return predict(root.children[instance_Feature_Value_Index],instance)
    

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    true_Predictions = 0
    for instace in range(dataset.shape[0]):
        if dataset[instace][-1] == predict(node , dataset[instace]):
            true_Predictions += 1
    accuracy = true_Predictions / dataset.shape[0] * 100
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(X_train , impurity=calc_entropy ,gain_ratio=True ,max_depth=max_depth)
        training.append(calc_accuracy(tree , X_train))
        testing.append(calc_accuracy(tree , X_test))
    return training, testing


def tree_Depth(root):
    if root.terminal:
        return 0
    return max([tree_Depth(child) for child in root.children]) + 1


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    for chi_Value in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = build_tree(X_train , impurity=calc_entropy ,gain_ratio=True ,chi=chi_Value)
        chi_training_acc.append(calc_accuracy(tree , X_train))
        chi_testing_acc.append(calc_accuracy(tree , X_test))
        depth.append(tree_Depth(tree))
    return chi_training_acc, chi_testing_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = 1
    if node.terminal:
        return n_nodes
    for child in node.children:
        n_nodes += count_nodes(child)
    return n_nodes






