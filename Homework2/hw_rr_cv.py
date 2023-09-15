import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load Data
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)


def RR(tau, CustomLambda, K):

    K_Errors = []
    for i in range(K):
        print(f"Fold {i + 1}:")

        # Determine the indices for the current fold
        fold_size = len(data) // K
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size


        # Split the data
        num_train = int(tau * fold_size)
        num_test = fold_size - num_train

        # Training Data copy within the current fold
        sample_train = data[start_idx:(end_idx-num_test), :-1]
        label_train = data[start_idx:(end_idx-num_test), -1]

        # Test data copy within the current fold
        sample_test = data[(start_idx+num_train):end_idx, :-1]
        label_test = data[(start_idx+num_train):end_idx, -1]

        # Training Data copy
        #sample_train = kportion[:num_train, :-1]
        #label_train = kportion[:num_train, -1]

        # Test data copy
        #sample_test = kportion[num_train:, :-1]
        #label_test = kportion[num_train:, -1]

        # Variables for calculation
        X = np.hstack((sample_train, np.ones((num_train, 1))))  # Add a column of ones for the intercept
        X_transpose = np.transpose(X)

        # Define your hyperparameter (alpha in Ridge regression)
        RidgePenalty = CustomLambda * np.identity(X.shape[1])
        RidgePenalty[0][0] = 0  # ignore the intercept variable

        # ACTUAL TRAINING
        beta = np.linalg.inv(X_transpose @ X + RidgePenalty) @ X_transpose @ label_train


        # Calculate predictions
        sample_train_with_intercept = np.hstack((sample_train, np.ones((num_train, 1))))
        train_prediction = sample_train_with_intercept @ beta

        #test model prediction
        sample_test_with_intercept = np.hstack((sample_test, np.ones((num_test, 1))))
        test_prediction = sample_test_with_intercept @ beta

        # Calculate the Test error with weighted squared differences and print
        error_test = mean_squared_error(label_test, test_prediction)
        print(f"Test Error: {error_test}")
        K_Errors.append(error_test)

    return K_Errors


# Pick 5 values of τ, including τ = 1
tau = 0.75
lam = 0.6
K = 10

K_Errors = RR(tau, lam, K)

print(f"K_Error average:{np.mean(K_Errors)}")