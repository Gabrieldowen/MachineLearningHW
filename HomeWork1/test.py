import numpy as np
import pandas as pd

# Load your dataset
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)

def WLS(tau, threshold):
    [n, p] = np.shape(data)

    # Split the data
    num_train = int(tau * n)
    num_test = n - num_train

    weights = np.ones(num_train)

    sample_train = data[:num_train, :-1]
    sample_test = data[num_train:, :-1]
    label_train = data[:num_train, -1]
    label_test = data[num_train:, -1]

    # Split data into high-risk and low-risk groups
    high_risk_communities = data[data[:, -1] > threshold]
    low_risk_communities = data[data[:, -1] <= threshold]

    # Get length of groups
    num_high_risk = len(high_risk_communities)
    num_low_risk = len(low_risk_communities)

    # Weights for risk groups
    wh = 10
    wl = 1

    # Define your hyperparameter (alpha in Ridge regression)
    alpha = 0.5

    # Implement Weighted Least Squares (WLS) from scratch
    # You need to calculate beta (the model coefficients) using the weights

    X = np.hstack((sample_train, np.ones((num_train, 1))))  # Add a column of ones for the intercept
    W = np.diag(weights)
    X_transpose = np.transpose(X)

    try:
        beta = np.linalg.inv(X_transpose @ W @ X + alpha * np.identity(p)) @ X_transpose @ W @ label_train
    except np.linalg.LinAlgError:
        print(f"Matrix inversion failed for tau={tau}.")
        return

    # Calculate predictions for high-risk and low-risk groups in testing set
    sample_test_with_intercept = np.hstack((sample_test, np.ones((num_test, 1))))
    label_test_pred = sample_test_with_intercept @ beta

    # Calculate the error with weighted squared differences
    # Error on high-risk group
    high_risk_indices = label_test > threshold
    error_high_risk = np.sqrt(np.sum((label_test[high_risk_indices] - label_test_pred[high_risk_indices]) ** 2) / num_test)

    # Error on low-risk group
    error_low_risk = np.sqrt(np.sum((label_test[~high_risk_indices] - label_test_pred[~high_risk_indices]) ** 2) / num_test)

    return error_high_risk, error_low_risk

# Set τ = 1 and threshold = 0.8
tau = 1
threshold = 0.8

# Calculate testing errors for high-risk and low-risk groups
error_high_risk, error_low_risk = WLS(tau, threshold)

# Create a table to report the testing errors
data = {'Group': ['High Risk', 'Low Risk'],
        'Testing Error': [error_high_risk, error_low_risk]}
df = pd.DataFrame(data)

print(df)
