###### Your ID ######
# ID1: 316317338
# ID2: 208597823
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    maxX = np.max(X ,axis=0)
    minX = np.min(X ,axis=0)
    meanX = np.mean(X ,axis=0)
    maxY = y.max()
    minY = y.min()
    meanY = y.mean()
    X = (X-meanX)/(maxX - minX)
    y = (y-meanY)/(maxY - minY)
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    X_new = np.column_stack((np.ones((X.shape[0],1)) , X))
    return X_new

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    M = X.shape[0]
    h_Theta = X.dot(theta)
    errors = h_Theta - y
    squared_Errors = np.square(errors)
    sigma_Squared_Errors = np.sum(squared_Errors)
    J = sigma_Squared_Errors/(2*M)  # We use J for the cost.
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    M = X.shape[0]
    for _ in range(num_iters):
        h_Theta = X.dot(theta)
        errors = h_Theta - y
        gradient = X.T.dot(errors) / M
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X,y,theta))
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    X = X.copy()
    X =  np.array(X)
    XT = X.transpose()
    XTX = XT.dot(X)
    pinvX = np.linalg.inv(XTX).dot(XT)
    pinv_theta = pinvX.dot(y) 
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    stop_condition = 1e-8
    M = X.shape[0]
    for i in range(num_iters):
        #check if the improvement is less than 1e-8 (if it's negative, the previous theta is better, return it instead)
        if i > 1:
            lossImprove = J_history[i-2]-J_history[i-1]
            if lossImprove < stop_condition:
                if lossImprove < 0:
                    theta = prevTheta
                break               
        h_Theta = X.dot(theta)
        errors = h_Theta - y
        gradient = X.T.dot(errors) / M
        prevTheta = theta
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X,y,theta))
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    np.random.seed(42)
    theta = np.random.random(X_train.shape[1])
    for alpha in alphas:
        tempTheta , _ = efficient_gradient_descent(X_train,y_train,theta,alpha,iterations)
        alpha_dict[alpha] = compute_cost(X_val , y_val , tempTheta)
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    best_Features_Matrix = np.ones((X_train.shape[0], 1))
    best_Features_Val_Matrix = np.ones((X_val.shape[0], 1))
    selected_features = []
    np.random.seed(42)
    for j in range(5):
        theta = np.random.random(size=j+2)
        min_Cost = np.inf
        Current_Best_Feature = -1
        for current_Feature_Index in range(X_train.shape[1]):
            if current_Feature_Index in selected_features:
                continue
            # add current feature to the matrices compute the cost and check if its better than the current best one
            best_Features_Matrix = np.hstack((best_Features_Matrix , X_train[:, current_Feature_Index][:, np.newaxis]))
            best_Features_Val_Matrix = np.hstack((best_Features_Val_Matrix , X_val[:, current_Feature_Index][:, np.newaxis]))
            current_Theta , _ = efficient_gradient_descent(best_Features_Matrix , y_train , theta ,best_alpha, iterations)
            current_Cost = compute_cost(best_Features_Val_Matrix,y_val,current_Theta)
            if min_Cost > current_Cost:
                min_Cost = current_Cost
                Current_Best_Feature = current_Feature_Index
            # delete the current feature from both matrices for the next iteration
            best_Features_Matrix = np.delete(best_Features_Matrix,j+1, axis=1)
            best_Features_Val_Matrix = np.delete(best_Features_Val_Matrix,j+1, axis=1)
        # eventally add the best feature found permenantly to both matrices and to the group
        selected_features.append(Current_Best_Feature)
        best_Features_Matrix = np.hstack((best_Features_Matrix,X_train[:, Current_Best_Feature][:, np.newaxis]))
        best_Features_Val_Matrix = np.hstack((best_Features_Val_Matrix,X_val[:, Current_Best_Feature][:, np.newaxis]))
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    for i , col_Name1 in enumerate(df.columns):
        for j , col_Name2 in enumerate(df.columns):
            # avoid duplicates
            if j >= i:
                if col_Name1 == col_Name2:
                    new_Col_Name = col_Name1 + '^2'
                else:
                    new_Col_Name = col_Name1 + '*' + col_Name2
                new_Col_Values = df[col_Name1] * df[col_Name2]
                new_Col_Values.rename(new_Col_Name, inplace=True)
                df_poly = pd.concat([df_poly , new_Col_Values], axis=1)
    return df_poly