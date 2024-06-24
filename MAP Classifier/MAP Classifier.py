import numpy as np
import math
class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0,
            (0, 1): 0.3,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.06,
            (0, 0, 1): 0.03,
            (0, 1, 0): 0.14,
            (0, 1, 1): 0.07,
            (1, 0, 0): 0.09,
            (1, 0, 1): 0.12,
            (1, 1, 0): 0.21,
            (1, 1, 1): 0.28,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        return not all(np.isclose(X[x] * Y[y] , X_Y[(x,y)]) for x in  X for y in Y)
    
    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        XgivenC = {(x,c) : (X_C[(x,c)] / C[c]) for x in X for c in C}
        YgivenC = {(y,c) : (Y_C[(y,c)] / C[c]) for y in Y for c in C}
        X_YgivenC = {(x,y,c) : (X_Y_C[(x,y,c)] / C[c]) for x in X for y in Y for c in C}
        return all(np.isclose(XgivenC[(x,c)] * YgivenC[(y,c)],X_YgivenC[(x,y,c)]) for x in  X for y in Y for c in C)
        

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    return  -rate + k * np.log(rate) - math.lgamma(k+1)

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = [np.sum([poisson_log_pmf(sample,rate) for sample in samples]) for rate in rates]

    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    rate = rates[np.argmax(likelihoods)]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    return np.mean(samples)

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = ((np.exp(np.square((x-mean)) / (-2*np.square(std))))/
            np.sqrt(2*np.pi*np.square(std)))
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        if class_value in dataset[:,-1]:
            indexes = dataset[:,-1] == class_value
            self.original_Num_Of_Instances = len(dataset)
            self.dataset = dataset[indexes]
            self.mean = np.mean(np.delete(self.dataset,-1,axis=1) ,axis=0)
            self.std = np.std(np.delete(self.dataset ,-1,axis=1), axis=0)

        else:
            print("no such class in the dataset")
            self = None
        
        
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.dataset) / self.original_Num_Of_Instances
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = np.prod([normal_pdf(x[i], self.mean[i], self.std[i]) for i in range(len(x))])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior()*self.get_instance_likelihood(x)
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.class0.get_instance_posterior(x) > self.class1.get_instance_posterior(x) else 1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    test_set_no_label = np.delete(test_set,-1,axis=1)
    correctly_Classified_Array = [(map_classifier.predict(test_set_no_label[instance]) == test_set[instance][-1])
                                    for instance in range(len(test_set))]
    correctly_Classified = np.sum(correctly_Classified_Array)
    acc = correctly_Classified / len(test_set)
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    root_det_Cov = np.power(np.linalg.det(cov),0.5)
    d = len(mean)
    inv_Cov = np.linalg.inv(cov)
    xMinusMeanTranpose = np.transpose(x - mean)
    pdf = ((np.exp(xMinusMeanTranpose @ inv_Cov @ (x-mean)/-2))/
           (root_det_Cov*np.sqrt(2*np.power(np.pi,d))))
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        if class_value in dataset[:,-1]:
            indexes = dataset[:,-1] == class_value
            self.original_Num_Of_Instances = len(dataset)
            self.dataset = dataset[indexes]
            self.mean = np.mean(np.delete(self.dataset,-1,axis=1) ,axis=0)
            self.cov = np.cov(np.delete(self.dataset ,-1,axis=1),rowvar=False)

        else:
            print("no such class in the dataset")
            self = None
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.dataset) / self.original_Num_Of_Instances
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x,self.mean,self.cov)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior()*self.get_instance_likelihood(x)
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self,x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.class0.get_prior() > self.class1.get_prior() else 1
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.class0.get_instance_likelihood(x) > self.class1.get_instance_likelihood(x) else 1
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        if class_value in dataset[:,-1]:
            self.class_value = class_value
            self.dataset = dataset
            self.class_dataset = dataset[self.dataset[:,-1] == self.class_value]
        else:
            print("no such class in the dataset")
            self = None

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = len(self.class_dataset) / len(self.dataset)
        return prior
    
    def calc_Laplace(self,x,j):
        """
        calculates P(xj|Ai) according to laplace formula
        x is the value of the j'th feature in the instnce X
        """
        if x in np.unique(self.dataset[:,j]): #check if Xj exists in the training set
            Vj = len(np.unique(self.dataset[:,j])) #the number of possible values in the j'th attribute

            #calculates the number of instances in class their j'th attribute equals x
            Xj_Indexes = self.class_dataset[:,j] == x 
            n_i_j = np.sum(Xj_Indexes)

            n_i = len(self.class_dataset) #number of instance in class 
            prob = (n_i_j + 1) / (n_i + Vj) #Laplace formula
        else:
            print("XJ")
            prob = EPSILLON
        return prob


    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = np.prod([self.calc_Laplace(x[j],j) for j in range(len(x))])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior()*self.get_instance_likelihood(x)
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.class0.get_instance_posterior(x) > self.class1.get_instance_posterior(x) else 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        test_set_no_label = np.delete(test_set,-1,axis=1)
        correctly_Classified_Array = [(self.predict(test_set_no_label[instance]) == test_set[instance][-1])
                                        for instance in range(len(test_set))]
        correctly_Classified = np.sum(correctly_Classified_Array)
        acc = correctly_Classified / len(test_set)
        return acc


