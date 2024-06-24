from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def preprocess_data( X , function ):
    return function(X) 

def standrtization(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def min_max_normalization(X):
    return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []


    def sigmoid(self , X):
        """
            Computes the sigmoid function
            Input:
            - X: m instances over n features where m can be 1 or more

            Returns:
            - h_Theta: m size array where each entry 'i' is the sigmoid of instance 'i' 
            
        """
        h_Theta = 1 / (np.exp(-X) + 1)
        return h_Theta
    
    def compute_cost(self, X, y):
        """
        Computes the Binary Cross Entropy cost function.  

        Input:
        - X: Input data (m instances over n features).
        - y: True labels (m instances).
        - theta: the parameters (weights) of the model being learned.

        Returns:
        - J: the cost associated with the current set of parameters (single number).
        """
        epsilon = 1e-10
        h_Theta = self.sigmoid(X @ self.theta)
        sigma =  ( -y * np.log(h_Theta + epsilon) ) - ((1 - y) * np.log(1 - h_Theta + epsilon)) 
        J = sigma.mean() # We use J for the cost.
        return J
    
    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        X_standrtization = standrtization(X)
        X_standrtization = np.column_stack((np.ones(X_standrtization.shape[0]), X_standrtization))   
        # set random seed
        np.random.seed(self.random_state)
        self.theta = np.random.rand(X_standrtization.shape[1])
        self.Js.append(self.compute_cost(X_standrtization,y))
        self.thetas.append(self.theta)
        for _ in range(self.n_iter):
            sigmoid = self.sigmoid(X_standrtization @ self.theta)
            delta_Tetha = - self.eta * X_standrtization.T @ (sigmoid - y)
            self.theta = self.theta  + delta_Tetha 
            cost = self.compute_cost(X_standrtization , y)
            if self.Js[-1] - cost < self.eps:
                if self.Js[-1] > cost:
                    self.Js.append(cost)
                    self.thetas.append(self.theta)
                break
            self.Js.append(cost)
            self.thetas.append(self.theta)

        self.theta = self.thetas[-1]
    

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X_standrtization = standrtization(X)
        X_standrtization = np.column_stack((np.ones(X_standrtization.shape[0]), X_standrtization))   
        return np.array([1 if self.sigmoid(X_standrtization[i] @ self.theta) >= 0.5 else 0 for i in range(len(X))])

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
    # set random seed
    np.random.seed(random_state)
    # Create shuffled indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    shuffled_X = X[indices] 
    shuffled_Y = y[indices]
    X_Folds_List = np.array_split(shuffled_X, folds)
    Y_Folds_List = np.array_split(shuffled_Y, folds)
    scores = []
    for i in range(folds):
      current_Validation_X = X_Folds_List[i]
      current_Validation_Y = Y_Folds_List[i]
      current_Train_X = np.concatenate(X_Folds_List[:i] + X_Folds_List[i + 1:])
      current_Train_Y = np.concatenate(Y_Folds_List[:i] + Y_Folds_List[i + 1:])
      algo.fit(current_Train_X , current_Train_Y)
      predicted_Y = algo.predict(current_Validation_X)
      scores.append(calc_Accuracy(current_Validation_Y,predicted_Y))
    return np.mean(scores)

def calc_Accuracy(y , predicted_Y):
    return np.mean(y == predicted_Y)




def norm_pdf(x, mu, sigma):
    """
    Calculate normal desnity function for a given x,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = (np.exp(np.square((x - mu)) / (-2 * np.square(sigma)))) / (np.sqrt(2 * np.pi * np.square(sigma))) 
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.mus = data[indices].reshape(self.k)
        self.sigmas = np.random.randint(1, 2 , self.k)
        self.weights = np.ones(self.k) / self.k
        self.costs = []

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        W_N_X = self.weights * norm_pdf(data , self.mus , self.sigmas)
        sum = np.sum(W_N_X ,axis=1 , keepdims=True)
        self.responsibilities = W_N_X / sum




    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.weights = np.mean(self.responsibilities, axis=0)
        self.mus = np.sum(self.responsibilities * data.reshape(-1,1), axis=0) / np.sum(self.responsibilities, axis=0)
        nominator = np.mean(self.responsibilities * np.square(data.reshape(-1, 1) - self.mus), axis=0)
        self.sigmas = np.sqrt(nominator  / self.weights)


    def log_Likelihood_Cost(self, data):
        W_N_X = norm_pdf(data , self.mus , self.sigmas) * self.weights
        W_N_X_Sum = np.sum(W_N_X , axis=1)
        return -np.sum(np.log(W_N_X_Sum))


    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        self.costs.append(self.log_Likelihood_Cost(data))
        for _ in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            cost = self.log_Likelihood_Cost(data)
            if self.costs[-1] - cost < self.eps:
                if self.costs[-1] > cost:
                    self.costs.append(cost)
                    break
            self.costs.append(cost)
  
    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = np.sum(weights * norm_pdf(data.reshape(-1,1) , mus , sigmas), axis=1)
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        
        self.X = X
        self.y = y
        self.num_Of_Instances = len(X)
        self.priors = {class_Label : len(y[y==class_Label])/len(y) for class_Label in np.unique(y)}
        self.gaussians = {class_Label: {feature : EM(self.k) for feature in range(X.shape[1])} for class_Label in np.unique(y)}
        for label in self.gaussians.keys():
            for feature in self.gaussians[label].keys():
              self.gaussians[label][feature].fit(X[y==label][:,feature].reshape(-1,1))

    def calc_Prior(self , class_label):
        return self.priors[class_label]

    def calc_likelihood(self , X, class_label):
        likelihhod = 1
        for feature in range(X.shape[0]):
          weights, mus, sigmas = self.gaussians[class_label][feature].get_dist_params()
          gmm = gmm_pdf(X[feature] , weights, mus, sigmas)
          likelihhod *= gmm
        return likelihhod
            
    def calc_posterior(self , X, class_label):
        return self.calc_Prior(class_label) * self.calc_likelihood(X, class_label)


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = [max([(self.calc_posterior(instance, class_Label),class_Label) for class_Label in self.priors.keys()] , key=lambda t:t[0])[1] for instance in X]
        return np.array(preds)

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    lor_train_acc = calc_Accuracy(y_train , lor.predict(x_train))
    lor_test_acc = calc_Accuracy(y_test , lor.predict(x_test))
    bayes = NaiveBayesGaussian(k=k)
    bayes.fit(x_train, y_train)
    bayes_train_acc = calc_Accuracy(y_train , bayes.predict(x_train))
    bayes_test_acc = calc_Accuracy(y_test , bayes.predict(x_test))
    if x_train.shape[1] == 2 :
      plot_decision_regions(x_train, y_train, lor)
      plot_decision_regions(x_train, y_train, bayes)
      #plot the cost per itrations in the lor model
      plt.plot(lor.Js)
      plt.title("Logistic Regression Cost")
      plt.xlabel("Iterations")
      plt.ylabel("Cost")
      plt.show()

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    np.random.seed(1991)
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    def generate_data(num_of_instances , mus, cov, labels):
        dataset_features = np.empty((num_of_instances, 3))
        dataset_labels = np.empty((num_of_instances))
        gaussian_size = num_of_instances//len(mus)

        for i , mu in enumerate(mus):
            label = labels[i]
            points = np.random.multivariate_normal(mu, cov, gaussian_size)
            dataset_features[i*gaussian_size: (i+1)* gaussian_size] = points
            dataset_labels[i * gaussian_size : (i+1) * gaussian_size] = np.full(gaussian_size , label)
        return dataset_features, dataset_labels
    
    # create a function that plots the 3d features and give color to each label
    def plot_data(features, labels, title):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels , cmap='winter', s=10, alpha=0.5, edgecolors='black', linewidths=0.5)
        ax.set_title(title)
        plt.show()

    dataset_a_mus = [[0, 0, 0], [4, 4, 4], [12, 12, 12], [18, 18, 18]]
    dataset_a_cov = [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]
    a_labels = [1, 0, 0, 1]

    dataset_a_features, dataset_a_labels = generate_data(5000 ,dataset_a_mus, dataset_a_cov, a_labels)
    plot_data(dataset_a_features, dataset_a_labels, "Dataset A - Naive Bayes is better")

    b_mus = [[0, 5, 0], [0, 7, 0]]  # Decrease the separation between means
    b_cov = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]  # Decrease the covariance to make the classes more separable
    b_labels = [0, 1]

    dataset_b_features, dataset_b_labels = generate_data(5000 ,b_mus, b_cov, b_labels)
    plot_data(dataset_b_features, dataset_b_labels,
              "Dataset B - LOR is better")

    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.show()