import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load Data
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)


def RR(tau, CustomLambda):
    [n, p] = np.shape(data)

    # Split the data
    num_train = int(tau * n)
    num_test = n - num_train

    # Training Data copy
    sample_train = data[:num_train, :-1]
    label_train = data[:num_train, -1]

    # Test data copy
    sample_test = data[num_train:, :-1]
    label_test = data[num_train:, -1]

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
    return error_test


RR(0.75, 1)

# Pick 5 values of τ, including τ = 1
lambda_values = [0.0001, 0.1, 0.5, 0.75, 1]
tau = 0.75

# Calculate errors for each lambda value
testing_errors = []

for lmbda in lambda_values:
    testing_errors.append(RR(tau, lmbda))


# Create a graph to report both training and testing errors versus τ
plt.figure(figsize=(8, 6))
plt.plot(lambda_values, testing_errors, marker='o', label='Testing Error')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.title('Testing Error versus lambda')
plt.legend()
plt.grid(True)
plt.show()